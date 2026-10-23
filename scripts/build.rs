//! Build script and standalone model downloader for Transcriber CLI.
//!
//! Checks if required ONNX models exist in `models/basic` and `models/streaming`.
//! If they are already present, exits immediately with zero network access.
//! If missing, downloads them once directly from official Hugging Face repositories.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct ModelFile {
    rel_path: &'static str,
    url: &'static str,
    min_size_bytes: u64,
}

const BASIC_MODELS: &[ModelFile] = &[
    ModelFile {
        rel_path: "basic/vocab.txt",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
        min_size_bytes: 50_000,
    },
    ModelFile {
        rel_path: "basic/decoder_joint-model.int8.onnx",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx",
        min_size_bytes: 10_000_000,
    },
    ModelFile {
        rel_path: "basic/encoder-model.int8.onnx",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx",
        min_size_bytes: 500_000_000,
    },
];

const STREAMING_MODELS: &[ModelFile] = &[
    ModelFile {
        rel_path: "streaming/config.json",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/config.json",
        min_size_bytes: 1_000,
    },
    ModelFile {
        rel_path: "streaming/tokenizer.model",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/tokenizer.model",
        min_size_bytes: 200_000,
    },
    ModelFile {
        rel_path: "streaming/decoder_joint.onnx",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/decoder_joint.onnx",
        min_size_bytes: 15_000_000,
    },
    ModelFile {
        rel_path: "streaming/encoder.onnx",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/encoder.onnx",
        min_size_bytes: 25_000_000,
    },
    ModelFile {
        rel_path: "streaming/encoder.onnx.data",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/encoder.onnx.data",
        min_size_bytes: 400_000_000,
    },
];

/// Locates the base `models/` directory in the repository.
fn find_models_dir() -> PathBuf {
    // 1. CARGO_MANIFEST_DIR is set during `cargo build`
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest).join("models");
        return p;
    }

    // 2. Search upwards from current directory
    let mut current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..5 {
        let candidate = current.join("models");
        if candidate.exists() {
            return candidate;
        }
        if current.join("Cargo.toml").exists() {
            return current.join("models");
        }
        if !current.pop() {
            break;
        }
    }

    PathBuf::from("models")
}

/// Checks if a file exists and satisfies the minimum expected size.
fn is_valid_file(path: &Path, min_size: u64) -> bool {
    if !path.exists() {
        return false;
    }
    match fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.len() >= min_size,
        Err(_) => false,
    }
}

/// Downloads a file from `url` to `dest_path`.
/// Uses `curl` with resume and progress support.
fn download_file(url: &str, dest_path: &Path) -> Result<(), String> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    let temp_path = dest_path.with_extension("download.tmp");

    eprintln!(
        "Downloading {} ...\n  URL: {}",
        dest_path.file_name().unwrap_or_default().to_string_lossy(),
        url
    );

    // Invoke curl
    let status = Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("--fail") // Fail on HTTP 4xx/5xx
        .arg("--progress-bar") // Clean progress display
        .arg("-C") // Resume if partially downloaded
        .arg("-")
        .arg("-o")
        .arg(&temp_path)
        .arg(url)
        .status()
        .map_err(|e| {
            format!(
                "Failed to execute curl: {}. Please ensure 'curl' is installed.",
                e
            )
        })?;

    if !status.success() {
        return Err(format!(
            "curl download failed with exit code: {:?}",
            status.code()
        ));
    }

    // Atomically rename temporary file to destination
    fs::rename(&temp_path, dest_path).map_err(|e| {
        format!(
            "Failed to rename {} to {}: {}",
            temp_path.display(),
            dest_path.display(),
            e
        )
    })?;

    eprintln!(
        "Downloaded: {}",
        dest_path.file_name().unwrap_or_default().to_string_lossy()
    );

    Ok(())
}

fn check_and_download_group(models_dir: &Path, files: &[ModelFile], group_name: &str) -> Result<(), String> {
    let mut missing_count = 0;

    for model in files {
        let dest = models_dir.join(model.rel_path);
        if !is_valid_file(&dest, model.min_size_bytes) {
            missing_count += 1;
        }
    }

    if missing_count == 0 {
        eprintln!("models/{}: verified.", group_name);
        return Ok(());
    }

    eprintln!(
        "models/{}: {} file(s) missing or incomplete. Downloading...",
        group_name, missing_count
    );

    for model in files {
        let dest = models_dir.join(model.rel_path);
        if !is_valid_file(&dest, model.min_size_bytes) {
            download_file(model.url, &dest)?;
        }
    }

    Ok(())
}

pub fn main() {
    // Tell Cargo to re-run this build script only if the models directory changes
    println!("cargo:rerun-if-changed=models");

    let models_dir = find_models_dir();
    fs::create_dir_all(&models_dir).ok();

    eprintln!("Checking models in {}...", models_dir.display());

    if let Err(e) = check_and_download_group(&models_dir, BASIC_MODELS, "basic") {
        eprintln!("error: failed to prepare basic models: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = check_and_download_group(&models_dir, STREAMING_MODELS, "streaming") {
        eprintln!("error: failed to prepare streaming models: {}", e);
        std::process::exit(1);
    }

    eprintln!("All models verified in models/.\n");
}
