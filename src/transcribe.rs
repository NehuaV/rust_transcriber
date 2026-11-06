//! Offline speech-to-text for macOS.
//!
//! Decodes any audio or video file with statically linked FFmpeg, then runs
//! NVIDIA Parakeet TDT 0.6B v3 on Apple CoreML. The pieces:
//!
//! - [`audio`]: decode a file to 16 kHz mono samples.
//! - [`models`]: find the model files on disk.
//! - [`window`]: cut a recording into overlapping windows and stitch words back.
//! - [`coreml`]: the CoreML models and the TDT decoding loop.
//! - [`pipeline`]: decode every window several ways on worker threads.
//! - [`word`]: words with timestamps and their text operations.

mod audio;
mod coreml;
mod models;
mod pipeline;
mod window;
mod word;

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::audio::DecodedAudio;

/// Command line arguments.
#[derive(Parser, Debug)]
#[command(
    name = "transcribe",
    author = "Dobri",
    version = "0.1.0",
    about = "Audio transcription CLI powered by NVIDIA Parakeet TDT 0.6B on CoreML and static FFmpeg",
    long_about = "Transcribe audio and video containers (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, WEBM, MP4, etc.) using local NVIDIA Parakeet TDT weights on Apple CoreML and statically linked FFmpeg decoding."
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
        default_value_t = pipeline::DEFAULT_VARIANTS,
        value_parser = clap::value_parser!(u8).range(1..=pipeline::VARIANTS.len() as i64)
    )]
    variants: u8,
}

/// Status line shown while working, or nothing in quiet mode.
struct Status(Option<ProgressBar>);

impl Status {
    /// A spinner on stderr, unless `quiet`.
    fn new(quiet: bool) -> Self {
        if quiet {
            return Self(None);
        }
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("-\\|/")
                .template("{spinner} {msg}")
                .expect("the spinner template is valid"),
        );
        spinner.enable_steady_tick(Duration::from_millis(80));
        Self(Some(spinner))
    }

    /// Shows `message` next to the spinner.
    fn message(&self, message: impl Into<std::borrow::Cow<'static, str>>) {
        if let Some(spinner) = &self.0 {
            spinner.set_message(message);
        }
    }

    /// Removes the spinner.
    fn finish(&self) {
        if let Some(spinner) = &self.0 {
            spinner.finish_and_clear();
        }
    }
}

fn main() -> Result<()> {
    run(&Cli::parse())
}

/// Transcribes the input file and prints, and optionally saves, the result.
fn run(cli: &Cli) -> Result<()> {
    if !cli.input.exists() {
        bail!("Input audio file does not exist: {}", cli.input.display());
    }

    let status = Status::new(cli.quiet);
    status.message("Decoding audio...");
    let audio = audio::decode(&cli.input)
        .context("Failed to decode audio. Please ensure the file contains a valid audio stream.")?;

    status.message("Loading Parakeet TDT CoreML models...");
    let started = Instant::now();
    let model_dir = models::coreml_dir()?;
    let text = pipeline::transcribe(
        &audio.samples,
        &model_dir,
        usize::from(cli.variants),
        &|done, total| status.message(format!("Transcribing window {done}/{total}...")),
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    status.finish();

    let text = text.trim();
    if cli.quiet {
        println!("{text}");
    } else {
        print_report(cli, &audio, text, elapsed);
    }

    if let Some(path) = &cli.output {
        fs::write(path, text)
            .with_context(|| format!("Failed to write transcript to {}", path.display()))?;
        if !cli.quiet {
            println!("\nSaved transcript to: {}", path.display());
        }
    }
    Ok(())
}

/// Prints the input summary, the speed and the transcript.
fn print_report(cli: &Cli, audio: &DecodedAudio, text: &str, elapsed_secs: f64) {
    let filename = cli.input.file_name().unwrap_or_default().to_string_lossy();
    let speed = if elapsed_secs > 0.0 && audio.duration_secs > 0.0 {
        (audio.duration_secs / elapsed_secs * 10.0).round() / 10.0
    } else {
        0.0
    };

    println!(
        "Input: {filename} (duration: {}, codec: {}, size: {})",
        format_duration(audio.duration_secs),
        audio.codec_name,
        format_bytes(audio.size_bytes)
    );
    println!(
        "Speed: {speed:.1}x real-time ({elapsed_secs:.2}s elapsed, {} words)",
        text.split_whitespace().count()
    );
    println!();
    if text.is_empty() {
        println!("(No speech detected)");
    } else {
        println!("{text}");
    }
}

/// Formats seconds as `MM:SS`.
fn format_duration(seconds: f64) -> String {
    let total = seconds.round() as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Formats a byte count as B, KB or MB.
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let value = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if value < KB * KB {
        format!("{:.1} KB", value / KB)
    } else {
        format!("{:.1} MB", value / (KB * KB))
    }
}

#[cfg(test)]
#[path = "transcribe_tests.rs"]
mod tests;
