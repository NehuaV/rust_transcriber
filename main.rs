use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, Transcriber};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_MODEL_DIR: &str = ".cache/parakeet-rs/parakeet-tdt-0.6b-v3";
const HF_BASE_URL: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";

#[derive(Parser, Debug)]
#[command(
    name = "transcriber",
    author = "Dobri",
    version = "0.1.0",
    about = "⚡ Pure Rust audio transcriber using NVIDIA Parakeet TDT 0.6B v3 via parakeet-rs",
    long_about = "Transcribe any audio or video format (M4A, MP3, WAV, FLAC, OGG, AAC, OPUS, WEBM, MP4, etc.) using parakeet-rs and NVIDIA Parakeet TDT 0.6B v3."
)]
struct Cli {
    /// Path to the audio or video file to transcribe
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Optional path to save the transcript text
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,

    /// Optional path to model directory containing ONNX weights
    #[arg(long, value_name = "MODEL_DIR")]
    model_dir: Option<PathBuf>,

    /// Output full transcription result as JSON
    #[arg(long)]
    json: bool,

    /// Only output the transcribed text, suppressing banners and metrics
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptionJson {
    text: String,
    duration: f64,
    processing_time: f64,
    rtfx: f64,
    model: String,
    word_count: usize,
    char_count: usize,
}

struct AudioInfo {
    duration: f64,
    codec: String,
    size_bytes: u64,
}

/// Inspect audio file with ffprobe to get duration, codec, and size
fn inspect_audio(input_path: &Path) -> AudioInfo {
    let size_bytes = fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
    let default_codec = input_path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let output = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name",
            "-of",
            "json",
            input_path.to_str().unwrap_or_default(),
        ])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            return AudioInfo {
                duration: 0.0,
                codec: default_codec,
                size_bytes,
            };
        }
    };

    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
        let duration = val
            .get("format")
            .and_then(|f| f.get("duration"))
            .and_then(|d| d.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let codec = val
            .get("streams")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|stream| stream.get("codec_name"))
            .and_then(|c| c.as_str())
            .unwrap_or(&default_codec)
            .to_string();

        AudioInfo {
            duration,
            codec,
            size_bytes,
        }
    } else {
        AudioInfo {
            duration: 0.0,
            codec: default_codec,
            size_bytes,
        }
    }
}

/// Convert any input audio/video format to 16kHz mono WAV using ffmpeg
fn convert_to_wav(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input.to_str().unwrap(),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .context("Failed to run ffmpeg. Please ensure ffmpeg is installed and on your PATH.")?;

    if !status.success() {
        bail!("FFmpeg failed to convert input audio to 16kHz WAV format");
    }
    Ok(())
}

/// Ensure model directory exists and has all required files
fn resolve_model_dir(custom_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = custom_path {
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }

    if let Ok(env_dir) = std::env::var("PARAKEET_MODEL_DIR") {
        let p = PathBuf::from(env_dir);
        if p.exists() {
            return Ok(p);
        }
    }

    let home = std::env::var("HOME").context("Could not determine HOME directory")?;
    let cache_dir = PathBuf::from(home).join(DEFAULT_MODEL_DIR);

    let vocab = cache_dir.join("vocab.txt");
    let encoder = cache_dir.join("encoder-model.int8.onnx");
    let decoder = cache_dir.join("decoder_joint-model.int8.onnx");

    if vocab.exists() && encoder.exists() && decoder.exists() {
        return Ok(cache_dir);
    }

    // Auto-download missing files if needed
    fs::create_dir_all(&cache_dir).context("Failed to create model cache directory")?;

    let files = [
        ("vocab.txt", vocab),
        ("decoder_joint-model.int8.onnx", decoder),
        ("encoder-model.int8.onnx", encoder),
    ];

    println!("{}", "📥 Downloading Parakeet TDT ONNX weights...".cyan());
    for (name, path) in &files {
        if !path.exists() {
            println!("  Downloading {}...", name.yellow());
            let url = format!("{}/{}", HF_BASE_URL, name);
            let status = Command::new("curl")
                .args(["-L", "-o", path.to_str().unwrap(), &url])
                .status()
                .with_context(|| format!("Failed to download {}", name))?;

            if !status.success() {
                bail!("Failed to download model file {} from {}", name, url);
            }
        }
    }

    Ok(cache_dir)
}

fn format_duration(seconds: f64) -> String {
    let total_secs = seconds.round() as u64;
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}", mins, secs)
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !cli.input.exists() {
        bail!("Input audio file does not exist: {}", cli.input.display());
    }

    let audio_info = inspect_audio(&cli.input);
    let input_filename = cli
        .input
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    if !cli.quiet && !cli.json {
        println!();
        println!(
            "{} {}",
            "🦜 PARAKEET TRANSCRIBER".bold().green(),
            "[parakeet-rs / NVIDIA Parakeet TDT 0.6B v3]".dimmed()
        );
        println!(
            "🎵 Input: {} ({}, {}, {})",
            input_filename.bold().white(),
            format_duration(audio_info.duration).cyan(),
            audio_info.codec.to_uppercase().yellow(),
            format_bytes(audio_info.size_bytes).dimmed()
        );
    }

    // Step 1: Resolve model directory
    let model_dir = resolve_model_dir(cli.model_dir.as_deref())?;

    // Terminal spinner
    let spinner = if !cli.quiet && !cli.json {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(Duration::from_millis(80));
        sp.set_message("Converting audio to 16kHz WAV format with FFmpeg...");
        Some(sp)
    } else {
        None
    };

    // Step 2: Convert any audio format to 16kHz mono WAV temp file
    let tmp_file = tempfile::Builder::new()
        .prefix("parakeet_")
        .suffix(".wav")
        .tempfile()
        .context("Failed to create temporary WAV file")?;
    let tmp_wav = tmp_file.path().to_path_buf();

    convert_to_wav(&cli.input, &tmp_wav)?;

    if let Some(ref sp) = spinner {
        sp.set_message("Transcribing with parakeet-rs (NVIDIA Parakeet TDT)...");
    }

    // Step 3: Run inference with parakeet-rs
    let t0 = Instant::now();
    let env = parakeet_rs::ort::init()
        .build()
        .context("Failed to initialize ONNX Runtime")?;
    let config = ExecutionConfig::new().with_environment(env);
    let mut model = ParakeetTDT::from_pretrained(&model_dir, Some(config))
        .context("Failed to load ParakeetTDT model in parakeet-rs")?;

    let result = model
        .transcribe_file(&tmp_wav, None)
        .context("Transcription failed in parakeet-rs")?;

    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(sp) = spinner {
        sp.finish_and_clear();
    }

    let text = result.text.trim().to_string();
    let duration = if audio_info.duration > 0.0 {
        audio_info.duration
    } else {
        // Fallback estimate if duration wasn't in metadata
        0.0
    };
    let rtfx = if elapsed > 0.0 && duration > 0.0 {
        (duration / elapsed * 10.0).round() / 10.0
    } else {
        0.0
    };
    let word_count = if text.is_empty() {
        0
    } else {
        text.split_whitespace().count()
    };
    let char_count = text.len();

    // Output formatting
    if cli.json {
        let json_data = TranscriptionJson {
            text: text.clone(),
            duration,
            processing_time: (elapsed * 1000.0).round() / 1000.0,
            rtfx,
            model: "nvidia/parakeet-tdt-0.6b-v3".to_string(),
            word_count,
            char_count,
        };
        println!("{}", serde_json::to_string_pretty(&json_data)?);
    } else if cli.quiet {
        println!("{}", text);
    } else {
        println!(
            "⚡ Engine: {} | Speed: {} (in {:.2}s) | Words: {}",
            "parakeet-rs (pure Rust)".magenta().bold(),
            format!("{:.1}x real-time", rtfx).bold().green(),
            elapsed,
            word_count.to_string().yellow()
        );
        println!("\n{}", "─".repeat(60).dimmed());
        println!("{}", "TRANSCRIPT:".bold().white());
        println!("{}", "─".repeat(60).dimmed());
        if text.is_empty() {
            println!("{}", "(No speech detected)".dimmed().italic());
        } else {
            println!("{}", text);
        }
        println!("{}\n", "─".repeat(60).dimmed());
    }

    // Save to output file if specified
    if let Some(ref out_path) = cli.output {
        if cli.json {
            let json_data = TranscriptionJson {
                text,
                duration,
                processing_time: (elapsed * 1000.0).round() / 1000.0,
                rtfx,
                model: "nvidia/parakeet-tdt-0.6b-v3".to_string(),
                word_count,
                char_count,
            };
            fs::write(out_path, serde_json::to_string_pretty(&json_data)?)
                .with_context(|| format!("Failed to write JSON output to {}", out_path.display()))?;
        } else {
            fs::write(out_path, &text)
                .with_context(|| format!("Failed to write transcript to {}", out_path.display()))?;
        }

        if !cli.quiet {
            println!(
                "{} {}",
                "💾 Saved transcript to:".bold().green(),
                out_path.display().to_string().cyan().underline()
            );
        }
    }

    Ok(())
}
