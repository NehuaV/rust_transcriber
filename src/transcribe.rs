use anyhow::{bail, Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

mod coreml;

use coreml::TimedToken;

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

    /// Ways each 15 second window is decoded; the fullest result is kept.
    /// More variants recover more text from windows the model leaves blank,
    /// at about 3.4 s per variant per 20 minutes of audio, but beyond 6 the
    /// model can also produce invented phrases.
    #[arg(
        short,
        long,
        value_name = "N",
        default_value_t = DEFAULT_VARIANTS,
        value_parser = clap::value_parser!(u8).range(1..=12)
    )]
    variants: u8,
}

/// Finds a model directory `models/<sub>` that contains `marker`.
/// Uses local disk weights without looking at .cache or making network requests.
fn find_model_dir(sub: &str, marker: &str, env_var: &str) -> Option<PathBuf> {
    let has_marker = |p: &Path| p.join(marker).exists();
    let rel = format!("models/{sub}");

    // 1. Check environment variable override
    if let Ok(dir_str) = std::env::var(env_var).or_else(|_| std::env::var("MODELS_DIR")) {
        let p = PathBuf::from(dir_str);
        let candidate = if has_marker(&p) { p } else { p.join(sub) };
        if has_marker(&candidate) {
            return Some(candidate);
        }
    }

    // 2. Check relative to current working directory
    let cwd_candidate = PathBuf::from(&rel);
    if has_marker(&cwd_candidate) {
        return Some(cwd_candidate);
    }

    // 3. Check relative to executable location (resolving symlinks)
    if let Ok(exe_path) = std::env::current_exe() {
        let real_exe = fs::canonicalize(&exe_path).unwrap_or(exe_path);
        if let Some(exe_dir) = real_exe.parent() {
            let candidates = [
                exe_dir.join(&rel),
                exe_dir.join("..").join(&rel),
                exe_dir.join("../share/transcriber").join(&rel),
            ];
            for candidate in candidates {
                if has_marker(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }

    // 4. Check user local share directory (~/.local/share/transcriber/models/<sub>)
    if let Ok(home) = std::env::var("HOME") {
        let share_dir = PathBuf::from(home)
            .join(".local/share/transcriber")
            .join(&rel);
        if has_marker(&share_dir) {
            return Some(share_dir);
        }
    }

    // 5. Compile-time directory fallback
    let manifest_rel = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&rel);
    if has_marker(&manifest_rel) {
        return Some(manifest_rel);
    }

    None
}

/// Resolves the CoreML model directory from `models/coreml`.
fn resolve_coreml_dir() -> Result<PathBuf> {
    find_model_dir("coreml", "parakeet_v3_vocab.json", "PARAKEET_COREML_DIR").context(
        "Parakeet CoreML models not found in 'models/coreml'.\n\
         Please run scripts/build.rs or execute:\n\
         cargo run --bin download-models",
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
fn absolute_keep_window(
    start: usize,
    end: usize,
    overlap: usize,
    is_first: bool,
    is_last: bool,
) -> (f32, f32) {
    let rate = SAMPLE_RATE as f32;
    let half_overlap = (overlap / 2) as f32 / rate;
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

/// Two words are the same word when their letters and digits match.
fn same_word(a: &str, b: &str) -> bool {
    let key = |w: &str| -> String {
        w.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let (a, b) = (key(a), key(b));
    !a.is_empty() && a == b
}

/// Seconds within which a repeated word across a window seam is the same word.
const SEAM_DUPLICATE_SECS: f32 = 0.25;

/// Concatenate the words each window owns, in order. Neighbouring windows time
/// the same word slightly differently, so a word at a seam can be kept by both.
/// A word that repeats the previous window's last word at (almost) the same
/// time is that duplicate and is dropped. A real repetition ("the the") is
/// spoken later than that and is kept.
fn merge_windows(windows: Vec<Vec<TimedToken>>) -> Vec<TimedToken> {
    let mut merged: Vec<TimedToken> = Vec::new();
    for window in windows {
        let seam = merged.len();
        for word in window {
            let duplicate = merged.len() == seam
                && merged.last().is_some_and(|prev| {
                    same_word(&prev.text, &word.text)
                        && (word.start - prev.start).abs() < SEAM_DUPLICATE_SECS
                });
            if !duplicate {
                merged.push(word);
            }
        }
    }
    merged
}

/// Join words with single spaces, attaching trailing punctuation to the previous word.
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

/// Keep only the words this chunk owns and shift their timestamps to the whole
/// recording. Trailing punctuation is folded onto the word it closes first.
fn own_words(
    tokens: Vec<TimedToken>,
    offset_samples: usize,
    start: usize,
    end: usize,
    overlap: usize,
    index: usize,
    total: usize,
) -> Vec<TimedToken> {
    let (keep_from, keep_to) =
        absolute_keep_window(start, end, overlap, index == 0, index + 1 == total);
    let offset = offset_samples as f32 / SAMPLE_RATE as f32;
    let mut kept = Vec::new();
    for mut word in fold_trailing_punctuation(tokens) {
        let mid = offset + (word.start + word.end) * 0.5;
        if mid >= keep_from && mid < keep_to && !word.text.trim().is_empty() {
            word.start += offset;
            word.end += offset;
            kept.push(word);
        }
    }
    kept
}

/// CoreML window length: the fixed-shape encoder takes exactly 15 seconds.
const COREML_WINDOW_SAMPLES: usize = coreml::WINDOW_SAMPLES;
/// Audio shared by neighbouring CoreML windows. Each cut sits in the middle of
/// the overlap, so every kept word had `overlap / 2` of context on both sides.
const COREML_OVERLAP_SAMPLES: usize = 4 * SAMPLE_RATE as usize;

/// Variants used unless `--variants` says otherwise.
const DEFAULT_VARIANTS: u8 = 6;

/// The ways a window is decoded: `(cut shift in seconds, length policy)`.
const VARIANTS: [(i64, coreml::LengthPolicy); 12] = [
    (0, coreml::LengthPolicy::ACTUAL),
    (0, coreml::LengthPolicy::TRIMMED_TAIL),
    (-1, coreml::LengthPolicy::ACTUAL),
    (0, coreml::LengthPolicy::ENCODER_FULL_TRIMMED),
    (1, coreml::LengthPolicy::ACTUAL),
    (0, coreml::LengthPolicy::PREPROCESSOR_FULL_TRIMMED),
    (-2, coreml::LengthPolicy::ACTUAL),
    (2, coreml::LengthPolicy::ACTUAL),
    (-1, coreml::LengthPolicy::TRIMMED_TAIL),
    (1, coreml::LengthPolicy::TRIMMED_TAIL),
    (-3, coreml::LengthPolicy::ACTUAL),
    (3, coreml::LengthPolicy::ACTUAL),
];

/// Decode one window `variants` ways (the first entries of `VARIANTS`) and keep
/// the result with the most words. Only the model's own outputs are compared;
/// nothing about the audio is assumed, and no word is edited. A variant that
/// blanks on this cut returns little or nothing, so it loses to one that does
/// not.
fn decode_window(
    engine: &coreml::CoreMlEngine,
    pcm: &[f32],
    variants: usize,
    start: usize,
    end: usize,
    index: usize,
    total: usize,
) -> Result<Vec<TimedToken>> {
    let rate = SAMPLE_RATE as i64;
    let mut best: Option<Vec<TimedToken>> = None;
    for &(shift, policy) in VARIANTS.iter().take(variants) {
        let span_start = (start as i64 + shift * rate).clamp(0, pcm.len() as i64) as usize;
        let span_end = (span_start + (end - start)).min(pcm.len());
        let nominal = span_start == start && span_end == end;
        if best.is_some() && shift != 0 && nominal {
            continue; // clamped to the same cut as the first variant
        }
        if span_end <= span_start {
            continue;
        }
        let words = engine
            .transcribe_window(&pcm[span_start..span_end], policy)
            .with_context(|| format!("CoreML inference failed (window {}/{})", index + 1, total))?;
        let owned = own_words(
            words,
            span_start,
            start,
            end,
            COREML_OVERLAP_SAMPLES,
            index,
            total,
        );
        if best.as_ref().is_none_or(|b| owned.len() > b.len()) {
            best = Some(owned);
        }
    }
    Ok(best.unwrap_or_default())
}

/// Most windows decoded at once. Each worker loads its own copy of the models
/// (they are memory-mapped, so this costs little RAM). Beyond about four the
/// Neural Engine, which runs the encoder one window at a time, is the limit.
const MAX_WINDOW_THREADS: usize = 4;

/// Worker threads for `windows` windows.
fn window_threads(windows: usize) -> usize {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    cores.min(MAX_WINDOW_THREADS).min(windows).max(1)
}

/// Transcribe 16 kHz mono PCM with CoreML. The encoder only takes 15 second
/// windows, so every file is windowed and stitched on word midpoints.
/// Transcribe 16 kHz mono PCM with CoreML. The encoder only takes 15 second
/// windows, so the whole recording is cut into overlapping windows, every one
/// is decoded as the model returns it, and the results are stitched on word
/// midpoints.
fn transcribe_audio(pcm: &[f32], variants: usize, spinner: Option<&ProgressBar>) -> Result<String> {
    if let Some(sp) = spinner {
        sp.set_message("Loading Parakeet TDT CoreML models...");
    }
    let dir = resolve_coreml_dir()
        .context("Failed to locate Parakeet CoreML models in 'models/coreml'")?;

    let spans = chunk_spans(pcm.len(), COREML_WINDOW_SAMPLES, COREML_OVERLAP_SAMPLES);
    let total = spans.len();
    let threads = window_threads(total);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Vec<TimedToken>>>> = Mutex::new(vec![None; total]);
    let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let engine = match coreml::CoreMlEngine::load(&dir) {
                    Ok(engine) => engine,
                    Err(e) => {
                        first_error.lock().unwrap().get_or_insert(e);
                        return;
                    }
                };
                loop {
                    if first_error.lock().unwrap().is_some() {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= total {
                        return;
                    }
                    let (start, end) = spans[i];
                    let decoded = decode_window(&engine, pcm, variants, start, end, i, total);
                    match decoded {
                        Ok(words) => {
                            slots.lock().unwrap()[i] = Some(words);
                            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if let Some(sp) = spinner {
                                sp.set_message(format!(
                                    "Transcribing window {}/{}...",
                                    finished, total
                                ));
                            }
                        }
                        Err(e) => {
                            first_error.lock().unwrap().get_or_insert(e);
                            return;
                        }
                    }
                }
            });
        }
    });

    if let Some(e) = first_error.into_inner().unwrap() {
        return Err(e);
    }
    let mut windows = Vec::with_capacity(total);
    for slot in slots.into_inner().unwrap() {
        windows.push(slot.context("Internal error: a window produced no result")?);
    }
    Ok(join_words(&merge_windows(windows)))
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

    // Step 2: Load the CoreML models and transcribe
    let t0 = Instant::now();
    let text = transcribe_audio(&audio.pcm_16k, cli.variants as usize, spinner.as_ref())?;
    let text = text.trim().to_string();

    let elapsed = t0.elapsed().as_secs_f64();
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

#[cfg(test)]
#[path = "coreml_tests.rs"]
mod coreml_tests;
