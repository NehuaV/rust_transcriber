use anyhow::{bail, Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use indicatif::{ProgressBar, ProgressStyle};
use parakeet_rs::{ExecutionConfig, Nemotron, NemotronMode};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{Duration, Instant};

// Model directory resolution for local models/streaming.

#[derive(Parser, Debug)]
#[command(
    name = "stranscribe",
    author = "Dobri",
    version = "0.2.0",
    about = "Streaming speech-to-text CLI powered by NVIDIA Nemotron Multilingual 3.5 0.6B and static FFmpeg",
    long_about = "Transcribe audio and video containers with live streaming FastConformer-RNNT inference using weights from models/streaming. Supports 40+ languages with constant memory consumption."
)]
struct Cli {
    /// Path to the audio or video file to transcribe
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Optional output file path to save the transcription
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,

    /// Target language code or locale (e.g. "en", "en-US", "es", "de", "fr", "it", "auto", etc.)
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Only output the transcribed text, suppressing metrics and status messages
    #[arg(short, long)]
    quiet: bool,
}

/// Resolves the model directory for Nemotron Multilingual 3.5 from `models/streaming`.
/// Uses local disk weights without looking at .cache or making network requests.
fn resolve_model_dir() -> Result<PathBuf> {
    // 1. Check environment variable override
    if let Ok(dir_str) = std::env::var("NEMOTRON_MODEL_DIR").or_else(|_| std::env::var("MODELS_DIR")) {
        let p = PathBuf::from(dir_str);
        let candidate = if p.join("config.json").exists() { p } else { p.join("streaming") };
        if candidate.join("config.json").exists() {
            return Ok(candidate);
        }
    }

    // 2. Check relative to current working directory
    let cwd_candidate = PathBuf::from("models/streaming");
    if cwd_candidate.join("config.json").exists() {
        return Ok(cwd_candidate);
    }

    // 3. Check relative to executable location (resolving symlinks)
    if let Ok(exe_path) = std::env::current_exe() {
        let real_exe = fs::canonicalize(&exe_path).unwrap_or(exe_path);
        if let Some(exe_dir) = real_exe.parent() {
            let candidates = [
                exe_dir.join("models/streaming"),
                exe_dir.join("../models/streaming"),
                exe_dir.join("../share/transcriber/models/streaming"),
            ];
            for candidate in &candidates {
                if candidate.join("config.json").exists() {
                    return Ok(candidate.clone());
                }
            }
        }
    }

    // 4. Check user local share directory (~/.local/share/transcriber/models/streaming)
    if let Ok(home) = std::env::var("HOME") {
        let share_dir = PathBuf::from(home).join(".local/share/transcriber/models/streaming");
        if share_dir.join("config.json").exists() {
            return Ok(share_dir);
        }
    }

    // 5. Compile-time directory fallback
    let manifest_rel = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/streaming");
    if manifest_rel.join("config.json").exists() {
        return Ok(manifest_rel);
    }

    bail!(
        "Nemotron Streaming model not found in 'models/streaming'.\n\
         Please run scripts/build.rs or execute:\n\
         cargo run --bin download-models"
    )
}

/// Normalizes common language names to Nemotron prompt language codes.
fn normalize_language(lang: &str) -> &str {
    match lang.trim().to_lowercase().as_str() {
        "en" | "english" => "en",
        "bg" | "bulgarian" => "bg",
        "es" | "spanish" => "es",
        "de" | "german" => "de",
        "fr" | "french" => "fr",
        "it" | "italian" => "it",
        "pt" | "portuguese" => "pt",
        "ru" | "russian" => "ru",
        "zh" | "chinese" | "zh-cn" => "zh-CN",
        "ja" | "japanese" => "ja",
        "ko" | "korean" => "ko",
        "tr" | "turkish" => "tr",
        "ar" | "arabic" => "ar",
        "hi" | "hindi" => "hi",
        "nl" | "dutch" => "nl",
        "pl" | "polish" => "pl",
        "sv" | "swedish" => "sv",
        "uk" | "ukrainian" => "uk",
        "auto" | "detect" => "auto",
        _ => lang.trim(),
    }
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

struct AudioMetadata {
    duration_secs: f64,
    samples_decoded: usize,
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

    let spinner = if !cli.quiet {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("-\\|/")
                .template("{spinner} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(Duration::from_millis(80));
        sp.set_message("Locating Nemotron model in models/streaming...");
        Some(sp)
    } else {
        None
    };

    // Step 1: Locate local model directory
    let model_dir =
        resolve_model_dir().context("Failed to locate Nemotron model in 'models/streaming'")?;

    if let Some(ref sp) = spinner {
        sp.set_message("Initializing Nemotron streaming engine...");
    }

    // Step 2: Initialize ONNX Runtime environment and Nemotron instance
    let env = parakeet_rs::ort::init()
        .build()
        .context("Failed to initialize ONNX Runtime environment")?;
    let config = ExecutionConfig::new().with_environment(env);
    let mut nemotron = Nemotron::from_pretrained(&model_dir, Some(config))
        .context("Failed to load Nemotron model from 'models/streaming'")?;

    let target_lang = normalize_language(&cli.language);
    if nemotron.mode() == NemotronMode::Multilingual {
        nemotron.set_target_lang(target_lang).with_context(|| {
            format!(
                "Failed to set Nemotron target language to '{}'",
                target_lang
            )
        })?;
    }

    // Step 3: Probe initial stream metadata before launching threads
    ffmpeg::init().context("Failed to initialize FFmpeg")?;
    unsafe {
        ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_ERROR);
    }

    let size_bytes = fs::metadata(&cli.input).map(|m| m.len()).unwrap_or(0);
    let probe_ctx = ffmpeg::format::input(&cli.input)
        .with_context(|| format!("Failed to open media file: {}", cli.input.display()))?;
    let probe_stream = probe_ctx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("Could not find playable audio stream in file")?;

    let initial_duration = if probe_stream.duration() > 0 {
        probe_stream.duration() as f64 * f64::from(probe_stream.time_base())
    } else if probe_ctx.duration() > 0 {
        probe_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
    } else {
        0.0
    };

    let probe_context =
        ffmpeg::codec::context::Context::from_parameters(probe_stream.parameters())?;
    let probe_decoder = probe_context.decoder().audio()?;
    let codec_name = probe_decoder
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    drop(probe_decoder);
    drop(probe_ctx);

    if let Some(sp) = spinner {
        sp.finish_and_clear();
    }

    // Print initial header if not in quiet mode
    if !cli.quiet {
        println!(
            "Input: {} (duration: {}, codec: {}, size: {})",
            input_filename,
            format_duration(initial_duration),
            codec_name,
            format_bytes(size_bytes)
        );
        println!("Language: {}", target_lang);
        println!();
    }

    // Step 4: Multithreaded Streaming Pipeline
    // Producer Thread: Decodes and resamples audio packets into a bounded channel
    let (tx, rx) = sync_channel::<Vec<f32>>(32);
    let input_path_clone = cli.input.clone();

    let producer_handle = thread::spawn(move || -> Result<AudioMetadata> {
        let mut ictx = ffmpeg::format::input(&input_path_clone).with_context(|| {
            format!("Failed to open media file: {}", input_path_clone.display())
        })?;

        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .context("Could not find playable audio stream in file")?;

        let stream_index = stream.index();
        let stream_duration = if stream.duration() > 0 {
            stream.duration() as f64 * f64::from(stream.time_base())
        } else if ictx.duration() > 0 {
            ictx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        };

        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        let mut decoder = context.decoder().audio()?;

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
        )?;

        let mut samples_decoded = 0usize;

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
                                samples_decoded += len;
                                if tx.send(slice.to_vec()).is_err() {
                                    break;
                                }
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
                    samples_decoded += len;
                    if tx.send(slice.to_vec()).is_err() {
                        break;
                    }
                }
            }
        }

        Ok(AudioMetadata {
            duration_secs: stream_duration,
            samples_decoded,
        })
    });

    // Consumer (Main Thread): Process audio chunks in real time and stream text to terminal
    let t0 = Instant::now();
    let mut is_first_token = true;

    while let Ok(chunk) = rx.recv() {
        let chunk_text = nemotron.transcribe_chunk(&chunk)?;
        if !chunk_text.is_empty() {
            let to_print = if is_first_token {
                is_first_token = false;
                chunk_text.trim_start()
            } else {
                &chunk_text
            };
            print!("{to_print}");
            let _ = io::stdout().flush();
        }
    }

    let elapsed_secs = t0.elapsed().as_secs_f64();
    let meta = producer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Producer thread panicked"))??;

    let final_duration = if meta.duration_secs > 0.0 {
        meta.duration_secs
    } else {
        meta.samples_decoded as f64 / 16000.0
    };

    let full_transcript = nemotron.get_transcript().trim().to_string();
    let word_count = if full_transcript.is_empty() {
        0
    } else {
        full_transcript.split_whitespace().count()
    };

    let rtfx = if elapsed_secs > 0.0 && final_duration > 0.0 {
        (final_duration / elapsed_secs * 10.0).round() / 10.0
    } else {
        0.0
    };

    if cli.quiet {
        // In quiet mode, ensure a trailing newline after live text
        println!();
    } else {
        println!();
        println!(
            "\nStreamed in {:.2}s ({:.1}x real-time, {} words)",
            elapsed_secs, rtfx, word_count
        );
    }

    // Save full transcript to output file if requested
    if let Some(ref out_path) = cli.output {
        fs::write(out_path, &full_transcript)
            .with_context(|| format!("Failed to write transcript to {}", out_path.display()))?;

        if !cli.quiet {
            println!("Saved transcript to: {}", out_path.display());
        }
    }

    Ok(())
}
