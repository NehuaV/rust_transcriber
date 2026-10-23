use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::*;
use ffmpeg_next as ffmpeg;
use indicatif::{ProgressBar, ProgressStyle};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, Transcriber};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ============================================================================
// EMBEDDED MODEL ASSETS (Baked directly into the compiled Rust binary)
// Zero reliance on external network downloads or system paths!
// ============================================================================
const VOCAB_BYTES: &[u8] = include_bytes!("../../models/vocab.txt");
const DECODER_BYTES: &[u8] = include_bytes!("../../models/decoder_joint-model.int8.onnx");
const ENCODER_BYTES: &[u8] = include_bytes!("../../models/encoder-model.int8.onnx");

#[derive(Parser, Debug)]
#[command(
    name = "transcriber",
    author = "Dobri",
    version = "0.1.0",
    about = "⚡ Completely self-contained speech-to-text CLI powered by embedded NVIDIA Parakeet TDT 0.6B v3 and statically linked FFmpeg",
    long_about = "Transcribe any audio or video format (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, WEBM, MP4, etc.) using embedded NVIDIA Parakeet TDT weights and statically linked FFmpeg audio decoding."
)]
struct Cli {
    /// Path to the audio or video file to transcribe
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Optional output file path to save the transcription
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,

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

/// Unpacks embedded model files to cache directory if not already present.
fn ensure_embedded_model() -> Result<PathBuf> {
    let target_dir = if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache/parakeet-rs/embedded-tdt-0.6b-v3")
    } else {
        std::env::temp_dir().join("parakeet-rs-embedded-tdt-0.6b-v3")
    };

    fs::create_dir_all(&target_dir).context("Failed to create embedded model directory")?;

    let vocab_path = target_dir.join("vocab.txt");
    let decoder_path = target_dir.join("decoder_joint-model.int8.onnx");
    let encoder_path = target_dir.join("encoder-model.int8.onnx");

    if !vocab_path.exists()
        || fs::metadata(&vocab_path).map(|m| m.len()).unwrap_or(0) != VOCAB_BYTES.len() as u64
    {
        fs::write(&vocab_path, VOCAB_BYTES)
            .context("Failed to write embedded vocabulary to cache")?;
    }

    if !decoder_path.exists()
        || fs::metadata(&decoder_path).map(|m| m.len()).unwrap_or(0) != DECODER_BYTES.len() as u64
    {
        fs::write(&decoder_path, DECODER_BYTES)
            .context("Failed to write embedded decoder ONNX model to cache")?;
    }

    if !encoder_path.exists()
        || fs::metadata(&encoder_path).map(|m| m.len()).unwrap_or(0) != ENCODER_BYTES.len() as u64
    {
        fs::write(&encoder_path, ENCODER_BYTES)
            .context("Failed to write embedded encoder ONNX model to cache")?;
    }

    Ok(target_dir)
}

struct DecodedAudio {
    pcm_16k: Vec<f32>,
    duration_secs: f64,
    codec_name: String,
    size_bytes: u64,
}

/// Decodes any audio or video container (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, MP4, etc.)
/// into a 16kHz mono PCM buffer using natively statically linked FFmpeg C libraries (ffmpeg-next).
/// Zero external dynamic libraries or spawned commands required!
fn decode_audio<P: AsRef<Path>>(path: P) -> Result<DecodedAudio> {
    let path = path.as_ref();
    let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    ffmpeg::init().context("Failed to initialize statically linked FFmpeg")?;
    unsafe {
        ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_ERROR);
    }

    let mut ictx = ffmpeg::format::input(&path)
        .with_context(|| format!("Failed to open media file: {}", path.display()))?;

    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("Could not find audio stream in file")?;

    let stream_index = input_stream.index();
    let duration_secs = if input_stream.duration() > 0 {
        input_stream.duration() as f64 * f64::from(input_stream.time_base())
    } else if ictx.duration() > 0 {
        ictx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
    } else {
        0.0
    };

    let context = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
        .context("Failed to get codec context from stream parameters")?;
    let mut decoder = context
        .decoder()
        .audio()
        .context("Failed to instantiate FFmpeg audio decoder")?;

    let codec_name = decoder
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let target_format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
    let target_layout = ffmpeg::channel_layout::ChannelLayout::MONO;
    let target_rate = 16000;

    let in_layout = if decoder.channel_layout().is_empty() {
        ffmpeg::channel_layout::ChannelLayout::default(decoder.channels() as i32)
    } else {
        decoder.channel_layout()
    };

    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        decoder.format(),
        in_layout,
        decoder.rate(),
        target_format,
        target_layout,
        target_rate,
    )
    .context("Failed to create FFmpeg audio resampler")?;

    let mut pcm_16k: Vec<f32> = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_index {
            if decoder.send_packet(&packet).is_ok() {
                let mut decoded_frame = ffmpeg::util::frame::audio::Audio::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    let mut resampled_frame = ffmpeg::util::frame::audio::Audio::empty();
                    if resampler.run(&decoded_frame, &mut resampled_frame).is_ok() {
                        let len = resampled_frame.samples();
                        if len > 0 {
                            let data = resampled_frame.data(0);
                            let ptr = data.as_ptr() as *const f32;
                            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                            pcm_16k.extend_from_slice(slice);
                        }
                    }
                }
            }
        }
    }

    let _ = decoder.send_eof();
    let mut decoded_frame = ffmpeg::util::frame::audio::Audio::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resampled_frame = ffmpeg::util::frame::audio::Audio::empty();
        if resampler.run(&decoded_frame, &mut resampled_frame).is_ok() {
            let len = resampled_frame.samples();
            if len > 0 {
                let data = resampled_frame.data(0);
                let ptr = data.as_ptr() as *const f32;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                pcm_16k.extend_from_slice(slice);
            }
        }
    }

    if pcm_16k.is_empty() {
        bail!("Could not decode any audio samples from file");
    }

    let calculated_duration = pcm_16k.len() as f64 / 16000.0;
    let final_duration = if duration_secs > 0.0 {
        duration_secs
    } else {
        calculated_duration
    };

    Ok(DecodedAudio {
        pcm_16k,
        duration_secs: final_duration,
        codec_name,
        size_bytes,
    })
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

    let input_filename = cli
        .input
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Spinner for feedback
    let spinner = if !cli.quiet && !cli.json {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(Duration::from_millis(80));
        sp.set_message("Decoding audio (statically linked FFmpeg engine)...");
        Some(sp)
    } else {
        None
    };

    // Step 1: Decode audio directly in memory
    let audio = decode_audio(&cli.input)
        .context("Failed to decode audio. Please ensure the file contains a valid audio stream.")?;

    if let Some(ref sp) = spinner {
        sp.set_message("Loading embedded NVIDIA Parakeet TDT model...");
    }

    // Step 2: Ensure embedded model is ready
    let model_dir = ensure_embedded_model()
        .context("Failed to prepare embedded Parakeet TDT model")?;

    // Step 3: Run inference via parakeet-rs
    let t0 = Instant::now();
    let env = parakeet_rs::ort::init()
        .build()
        .context("Failed to initialize ONNX Runtime environment")?;
    let config = ExecutionConfig::new().with_environment(env);
    let mut model = ParakeetTDT::from_pretrained(&model_dir, Some(config))
        .context("Failed to load ParakeetTDT model from embedded assets")?;

    if let Some(ref sp) = spinner {
        sp.set_message("Transcribing with embedded Parakeet TDT...");
    }

    let result = model
        .transcribe_samples(audio.pcm_16k, 16000, 1, None)
        .context("Transcription failed during inference")?;

    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(sp) = spinner {
        sp.finish_and_clear();
    }

    let text = result.text.trim().to_string();
    let rtfx = if elapsed > 0.0 && audio.duration_secs > 0.0 {
        (audio.duration_secs / elapsed * 10.0).round() / 10.0
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
            duration: (audio.duration_secs * 100.0).round() / 100.0,
            processing_time: (elapsed * 1000.0).round() / 1000.0,
            rtfx,
            model: "nvidia/parakeet-tdt-0.6b-v3 [embedded]".to_string(),
            word_count,
            char_count,
        };
        println!("{}", serde_json::to_string_pretty(&json_data)?);
    } else if cli.quiet {
        println!("{}", text);
    } else {
        println!();
        println!(
            "{} {}",
            "🦜 PARAKEET TRANSCRIBER".bold().green(),
            "[Embedded Model • Statically Linked FFmpeg • Zero System Dependencies]".dimmed()
        );
        println!(
            "🎵 Input: {} ({}, {}, {})",
            input_filename.bold().white(),
            format_duration(audio.duration_secs).cyan(),
            audio.codec_name.yellow(),
            format_bytes(audio.size_bytes).dimmed()
        );
        println!(
            "⚡ Engine: {} | Speed: {} (in {:.2}s) | Words: {}",
            "parakeet-rs".magenta().bold(),
            format!("{:.1}x real-time", rtfx).bold().green(),
            elapsed,
            word_count.to_string().yellow()
        );
        println!("\n{}", "─".repeat(60).dimmed());
        println!("{}", "TRANSCRIPTION:".bold().white());
        println!("{}", "─".repeat(60).dimmed());
        if text.is_empty() {
            println!("{}", "(No speech detected)".dimmed().italic());
        } else {
            println!("{}", text);
        }
        println!("{}\n", "─".repeat(60).dimmed());
    }

    // Save to output file if requested
    if let Some(ref out_path) = cli.output {
        if cli.json {
            let json_data = TranscriptionJson {
                text,
                duration: (audio.duration_secs * 100.0).round() / 100.0,
                processing_time: (elapsed * 1000.0).round() / 1000.0,
                rtfx,
                model: "nvidia/parakeet-tdt-0.6b-v3 [embedded]".to_string(),
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
