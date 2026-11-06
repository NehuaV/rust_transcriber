use anyhow::{bail, Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use indicatif::{ProgressBar, ProgressStyle};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimedToken, TimestampMode, Transcriber};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// Model directory resolution for local models/basic.

#[derive(Parser, Debug)]
#[command(
    name = "transcribe",
    author = "Dobri",
    version = "0.1.0",
    about = "Audio transcription CLI powered by NVIDIA Parakeet TDT 0.6B and static FFmpeg",
    long_about = "Transcribe audio and video containers (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, WEBM, MP4, etc.) using local NVIDIA Parakeet TDT weights and statically linked FFmpeg decoding."
)]
struct Cli {
    /// Path to the audio or video file to transcribe
    #[arg(value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Optional output file path to save the transcription
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,

    /// Only output the transcribed text, suppressing metrics and status messages
    #[arg(short, long)]
    quiet: bool,
}

/// Resolves the model directory for Parakeet TDT from `models/basic`.
/// Uses local disk weights without looking at .cache or making network requests.
fn resolve_model_dir() -> Result<PathBuf> {
    // 1. Check environment variable override
    if let Ok(dir_str) =
        std::env::var("PARAKEET_MODEL_DIR").or_else(|_| std::env::var("MODELS_DIR"))
    {
        let p = PathBuf::from(dir_str);
        let candidate = if p.join("vocab.txt").exists() {
            p
        } else {
            p.join("basic")
        };
        if candidate.join("vocab.txt").exists() {
            return Ok(candidate);
        }
    }

    // 2. Check relative to current working directory
    let cwd_candidate = PathBuf::from("models/basic");
    if cwd_candidate.join("vocab.txt").exists() {
        return Ok(cwd_candidate);
    }

    // 3. Check relative to executable location (resolving symlinks)
    if let Ok(exe_path) = std::env::current_exe() {
        let real_exe = fs::canonicalize(&exe_path).unwrap_or(exe_path);
        if let Some(exe_dir) = real_exe.parent() {
            let candidates = [
                exe_dir.join("models/basic"),
                exe_dir.join("../models/basic"),
                exe_dir.join("../share/transcriber/models/basic"),
            ];
            for candidate in &candidates {
                if candidate.join("vocab.txt").exists() {
                    return Ok(candidate.clone());
                }
            }
        }
    }

    // 4. Check user local share directory (~/.local/share/transcriber/models/basic)
    if let Ok(home) = std::env::var("HOME") {
        let share_dir = PathBuf::from(home).join(".local/share/transcriber/models/basic");
        if share_dir.join("vocab.txt").exists() {
            return Ok(share_dir);
        }
    }

    // 5. Compile-time directory fallback
    let manifest_rel = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/basic");
    if manifest_rel.join("vocab.txt").exists() {
        return Ok(manifest_rel);
    }

    bail!(
        "Parakeet TDT model not found in 'models/basic'.\n\
         Please run scripts/build.rs or execute:\n\
         cargo run --bin download-models"
    )
}

struct DecodedAudio {
    pcm_16k: Vec<f32>,
    duration_secs: f64,
    codec_name: String,
    size_bytes: u64,
}

/// Decodes any audio or video container (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, MP4, etc.)
/// into a 16kHz mono PCM buffer using natively statically linked FFmpeg C libraries (ffmpeg-next).
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

/// 16 kHz mono, matching the Parakeet TDT preprocessor.
const SAMPLE_RATE: u32 = 16_000;
/// Mel hop used by the TDT preprocessor. Mel frames are `samples / HOP_LENGTH`.
const HOP_LENGTH: usize = 160;
/// Encoder steps the exported relative-position table can attend over.
///
/// The table is `[1, 9999, 1024]`, sliced around a center of 5000. Attention
/// then adds that slice (`2 * steps - 1` wide) onto scores of the same width.
/// Past 5000 steps the slice stays 9999 wide and the add cannot broadcast.
const MAX_ENCODER_STEPS: i64 = 5_000;
/// 6_400_000 samples is exactly 5000 encoder steps. The next hop is 5001.
const MAX_SAMPLES: usize = 6_400_000;
/// Five minutes. Stays under the positional-embedding limit and matches the
/// chunk length recommended for this model.
const CHUNK_SAMPLES: usize = 5 * 60 * SAMPLE_RATE as usize;
/// Shared audio on each side of a cut so a word on the boundary is fully
/// inside the chunk that owns it.
const OVERLAP_SAMPLES: usize = 15 * SAMPLE_RATE as usize;

/// Encoder length for `samples` of 16 kHz audio: `1 + (mel_frames - 1) / 8`.
fn encoder_steps(samples: usize) -> i64 {
    let mel_frames = (samples / HOP_LENGTH) as i64;
    if mel_frames <= 0 {
        0
    } else {
        1 + (mel_frames - 1) / 8
    }
}

/// Half-open sample ranges of at most `chunk_len`, each overlapping the next
/// by `overlap` samples. The last range always runs through `total`.
fn chunk_spans(total: usize, chunk_len: usize, overlap: usize) -> Vec<(usize, usize)> {
    assert!(chunk_len > overlap, "chunk length must exceed overlap");
    if total == 0 {
        return Vec::new();
    }
    if total <= chunk_len {
        return vec![(0, total)];
    }

    let stride = chunk_len - overlap;
    let mut spans = Vec::new();
    let mut start = 0;
    while start < total {
        let end = (start + chunk_len).min(total);
        spans.push((start, end));
        if end == total {
            break;
        }
        start += stride;
    }
    spans
}

/// Absolute time window `[from, to)` whose words belong to this chunk.
///
/// The cut sits in the middle of the overlap. Each chunk extends `overlap / 2`
/// past that cut, so a word assigned by its midpoint was heard in full.
/// The last chunk keeps every remaining word, including timestamps that land
/// slightly past the final sample.
fn absolute_keep_window(start: usize, end: usize, is_first: bool, is_last: bool) -> (f32, f32) {
    let rate = SAMPLE_RATE as f32;
    let half_overlap = (OVERLAP_SAMPLES / 2) as f32 / rate;
    let from = if is_first {
        0.0
    } else {
        start as f32 / rate + half_overlap
    };
    let to = if is_last {
        f32::MAX
    } else {
        end as f32 / rate - half_overlap
    };
    (from, to)
}

fn is_trailing_punct(text: &str) -> bool {
    text.len() == 1
        && text
            .chars()
            .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'))
}

/// Glue trailing punctuation onto the previous word and keep that word's timing,
/// so a period that lands just past a chunk cut stays with the word it closes.
fn fold_trailing_punctuation(words: Vec<TimedToken>) -> Vec<TimedToken> {
    let mut out: Vec<TimedToken> = Vec::with_capacity(words.len());
    for word in words {
        if is_trailing_punct(&word.text) {
            if let Some(prev) = out.last_mut() {
                prev.text.push_str(&word.text);
                continue;
            }
        }
        out.push(word);
    }
    out
}

/// Join word tokens the same way `parakeet-rs` rebuilds `TimestampMode::Words`.
fn join_words(words: &[TimedToken]) -> String {
    let mut out = String::new();
    for (i, word) in words.iter().enumerate() {
        let text = word.text.as_str();
        if i > 0 && !is_trailing_punct(text) {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

/// Transcribe one window and return only the words this chunk owns, with
/// timestamps shifted to the whole recording.
fn transcribe_chunk(
    model: &mut ParakeetTDT,
    pcm: &[f32],
    index: usize,
    total: usize,
    start: usize,
    end: usize,
) -> Result<Vec<TimedToken>> {
    let steps = encoder_steps(end - start);
    if steps > MAX_ENCODER_STEPS {
        bail!(
            "Internal error: chunk of {} samples is {} encoder steps, above the {} step limit",
            end - start,
            steps,
            MAX_ENCODER_STEPS
        );
    }
    let result = model
        .transcribe_samples(
            pcm[start..end].to_vec(),
            SAMPLE_RATE,
            1,
            Some(TimestampMode::Words),
        )
        .with_context(|| {
            format!(
                "Transcription failed during inference (chunk {}/{}, samples {}..{})",
                index + 1,
                total,
                start,
                end
            )
        })?;

    let (keep_from, keep_to) = absolute_keep_window(start, end, index == 0, index + 1 == total);
    let offset = start as f32 / SAMPLE_RATE as f32;
    let mut kept = Vec::new();
    for word in fold_trailing_punctuation(result.tokens) {
        let mid = offset + (word.start + word.end) * 0.5;
        if mid >= keep_from && mid < keep_to && !word.text.trim().is_empty() {
            kept.push(word);
        }
    }
    Ok(kept)
}

/// Transcribe 16 kHz mono PCM. Files that fit in the positional embedding run
/// in one pass. Longer files are cut into overlapping windows, transcribed one
/// at a time, and stitched on word midpoints so a word on a boundary is
/// emitted once.
fn transcribe_pcm(
    model: &mut ParakeetTDT,
    pcm: &[f32],
    spinner: Option<&ProgressBar>,
) -> Result<String> {
    if pcm.len() <= MAX_SAMPLES {
        if let Some(sp) = spinner {
            sp.set_message("Transcribing...");
        }
        let result = model
            .transcribe_samples(pcm.to_vec(), SAMPLE_RATE, 1, None)
            .context("Transcription failed during inference")?;
        return Ok(result.text);
    }

    let spans = chunk_spans(pcm.len(), CHUNK_SAMPLES, OVERLAP_SAMPLES);
    let total = spans.len();
    let mut kept = Vec::new();
    for (i, &(start, end)) in spans.iter().enumerate() {
        if let Some(sp) = spinner {
            sp.set_message(format!("Transcribing chunk {}/{}...", i + 1, total));
        }
        kept.extend(transcribe_chunk(model, pcm, i, total, start, end)?);
    }

    Ok(join_words(&kept))
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
    let spinner = if !cli.quiet {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("-\\|/")
                .template("{spinner} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(Duration::from_millis(80));
        sp.set_message("Decoding audio...");
        Some(sp)
    } else {
        None
    };

    // Step 1: Decode audio directly in memory
    let audio = decode_audio(&cli.input)
        .context("Failed to decode audio. Please ensure the file contains a valid audio stream.")?;

    if let Some(ref sp) = spinner {
        sp.set_message("Loading Parakeet TDT model...");
    }

    // Step 2: Locate local model in models/basic
    let model_dir =
        resolve_model_dir().context("Failed to locate Parakeet TDT model in 'models/basic'")?;

    // Step 3: Run inference via parakeet-rs
    let t0 = Instant::now();
    let env = parakeet_rs::ort::init()
        .build()
        .context("Failed to initialize ONNX Runtime environment")?;
    let config = ExecutionConfig::new().with_environment(env);
    let mut model = ParakeetTDT::from_pretrained(&model_dir, Some(config))
        .context("Failed to load ParakeetTDT model from 'models/basic'")?;

    let text = transcribe_pcm(&mut model, &audio.pcm_16k, spinner.as_ref())?
        .trim()
        .to_string();

    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(sp) = spinner {
        sp.finish_and_clear();
    }
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

    // Output formatting
    if cli.quiet {
        println!("{}", text);
    } else {
        println!(
            "Input: {} (duration: {}, codec: {}, size: {})",
            input_filename,
            format_duration(audio.duration_secs),
            audio.codec_name,
            format_bytes(audio.size_bytes)
        );
        println!(
            "Speed: {:.1}x real-time ({:.2}s elapsed, {} words)",
            rtfx, elapsed, word_count
        );
        println!();
        if text.is_empty() {
            println!("(No speech detected)");
        } else {
            println!("{}", text);
        }
    }

    // Save to output file if requested
    if let Some(ref out_path) = cli.output {
        fs::write(out_path, &text)
            .with_context(|| format!("Failed to write transcript to {}", out_path.display()))?;

        if !cli.quiet {
            println!("\nSaved transcript to: {}", out_path.display());
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
