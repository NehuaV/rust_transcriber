//! Build script and standalone model downloader for Transcriber CLI.
//!
//! Checks if required ONNX models exist in `models/basic` and `models/streaming`.
//! If they are already present and match exact sizes and standard library hashes, exits immediately.
//! If missing or corrupted, downloads them directly from official Hugging Face repositories.

use std::fs::{self, File};
use std::hash::{DefaultHasher, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

struct ModelFile {
    rel_path: &'static str,
    url: &'static str,
    expected_size: u64,
    expected_hash: u64,
}

const BASIC_MODELS: &[ModelFile] = &[
    ModelFile {
        rel_path: "basic/vocab.txt",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
        expected_size: 93_939,
        expected_hash: 0xc448_610b_3c14_a06e,
    },
    ModelFile {
        rel_path: "basic/decoder_joint-model.int8.onnx",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx",
        expected_size: 18_202_004,
        expected_hash: 0xab41_86fc_9cbb_0a76,
    },
    ModelFile {
        rel_path: "basic/encoder-model.int8.onnx",
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx",
        expected_size: 652_183_999,
        expected_hash: 0x5565_3a37_d4ab_d343,
    },
];

const STREAMING_MODELS: &[ModelFile] = &[
    ModelFile {
        rel_path: "streaming/config.json",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/config.json",
        expected_size: 2_970,
        expected_hash: 0xd726_4578_f18a_f3b8,
    },
    ModelFile {
        rel_path: "streaming/tokenizer.model",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/tokenizer.model",
        expected_size: 406_554,
        expected_hash: 0xb406_c155_b4e0_5d9d,
    },
    ModelFile {
        rel_path: "streaming/decoder_joint.onnx",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/decoder_joint.onnx",
        expected_size: 24_483_962,
        expected_hash: 0x12b0_b8b3_63b0_8d5c,
    },
    ModelFile {
        rel_path: "streaming/encoder.onnx",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/encoder.onnx",
        expected_size: 42_963_073,
        expected_hash: 0x9d53_86af_8885_9b04,
    },
    ModelFile {
        rel_path: "streaming/encoder.onnx.data",
        url: "https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8/resolve/main/encoder.onnx.data",
        expected_size: 614_649_600,
        expected_hash: 0x66c4_d97f_393a_046a,
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

/// Computes a hash of the file contents using standard library `std::hash::DefaultHasher`.
fn compute_std_hash(path: &Path) -> Result<u64, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let mut hasher = DefaultHasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        if bytes_read == 0 {
            break;
        }
        hasher.write(&buffer[..bytes_read]);
    }
    Ok(hasher.finish())
}

/// Checks if a file exists, matches exact expected size, and matches expected hash.
fn is_valid_file(path: &Path, expected_size: u64, expected_hash: u64) -> bool {
    if !path.exists() {
        return false;
    }
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() || meta.len() != expected_size {
        return false;
    }
    match compute_std_hash(path) {
        Ok(hash) => hash == expected_hash,
        Err(_) => false,
    }
}

/// Downloads a file cleanly from `url` to a unique temporary file,
/// verifies its size and standard hash, and atomically moves it to `dest_path`.
fn download_file(
    url: &str,
    dest_path: &Path,
    expected_size: u64,
    expected_hash: u64,
) -> Result<(), String> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    let pid = std::process::id();
    let file_name = dest_path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let temp_name = format!("{}.download.{}.tmp", file_name, pid);
    let temp_path = dest_path.with_file_name(temp_name);

    // Remove any leftover temporary file
    let _ = fs::remove_file(&temp_path);

    eprintln!("Downloading {} ...\n  URL: {}", file_name, url);

    let status = Command::new("curl")
        .arg("-L")              // Follow redirects
        .arg("--fail")          // Fail on HTTP 4xx/5xx
        .arg("--progress-bar")  // Clean progress display
        .arg("-o")
        .arg(&temp_path)
        .arg(url)
        .status()
        .map_err(|e| format!("Failed to execute curl: {}. Please ensure 'curl' is installed.", e))?;

    if !status.success() {
        let _ = fs::remove_file(&temp_path);
        return Err(format!("curl download failed with exit code: {:?}", status.code()));
    }

    // Verify downloaded file size
    let downloaded_size = match fs::metadata(&temp_path) {
        Ok(meta) => meta.len(),
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            return Err(format!("Failed to read metadata of {}: {}", temp_path.display(), e));
        }
    };

    if downloaded_size != expected_size {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "Size mismatch for {}: expected {} bytes, got {} bytes.",
            file_name, expected_size, downloaded_size
        ));
    }

    // Verify downloaded file hash using std hash function
    let downloaded_hash = match compute_std_hash(&temp_path) {
        Ok(h) => h,
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            return Err(format!("Failed to compute hash for {}: {}", temp_path.display(), e));
        }
    };

    if downloaded_hash != expected_hash {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "Hash mismatch for {}: expected {:016x}, got {:016x}.",
            file_name, expected_hash, downloaded_hash
        ));
    }

    // Atomically move to destination
    fs::rename(&temp_path, dest_path).map_err(|e| {
        let _ = fs::remove_file(&temp_path);
        format!(
            "Failed to rename {} to {}: {}",
            temp_path.display(),
            dest_path.display(),
            e
        )
    })?;

    eprintln!("Downloaded and verified: {} (hash: {:016x})", file_name, downloaded_hash);
    Ok(())
}

fn check_and_download_group(
    models_dir: &Path,
    files: &[ModelFile],
    group_name: &str,
) -> Result<(), String> {
    let mut missing = Vec::new();

    for model in files {
        let dest = models_dir.join(model.rel_path);
        if !is_valid_file(&dest, model.expected_size, model.expected_hash) {
            missing.push(model);
        }
    }

    if !missing.is_empty() {
        eprintln!(
            "models/{}: {} file(s) missing or incomplete. Downloading...",
            group_name,
            missing.len()
        );

        for model in &missing {
            let dest = models_dir.join(model.rel_path);
            download_file(
                model.url,
                &dest,
                model.expected_size,
                model.expected_hash,
            )?;
        }
    }

    eprintln!("models/{}: verified.", group_name);
    for model in files {
        let dest = models_dir.join(model.rel_path);
        let file_name = dest.file_name().unwrap_or_default().to_string_lossy();
        eprintln!("  - {}: ok (hash: {:016x})", file_name, model.expected_hash);
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
