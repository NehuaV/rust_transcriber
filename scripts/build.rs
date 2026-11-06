//! Build script and standalone model downloader for Transcriber CLI.
//!
//! Checks that the CoreML bundles in `models/coreml` exist.
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

/// Parakeet TDT 0.6B v3 CoreML export (FluidInference). `Encoder_v2` is the
/// int8 per-channel encoder that fixes window-edge word errors in the original.
const COREML_MODELS: &[ModelFile] = &[
    ModelFile {
        rel_path: "coreml/Decoder.mlmodelc/analytics/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Decoder.mlmodelc/analytics/coremldata.bin",
        expected_size: 243,
        expected_hash: 0x2332_3859_f576_5030,
    },
    ModelFile {
        rel_path: "coreml/Decoder.mlmodelc/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Decoder.mlmodelc/coremldata.bin",
        expected_size: 554,
        expected_hash: 0xa6a9_f8b2_b5a6_89b8,
    },
    ModelFile {
        rel_path: "coreml/Decoder.mlmodelc/metadata.json",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Decoder.mlmodelc/metadata.json",
        expected_size: 3_427,
        expected_hash: 0x1304_903a_e1b8_4cf6,
    },
    ModelFile {
        rel_path: "coreml/Decoder.mlmodelc/model.mil",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Decoder.mlmodelc/model.mil",
        expected_size: 13_110,
        expected_hash: 0xefc2_1a87_9f11_ba7e,
    },
    ModelFile {
        rel_path: "coreml/Decoder.mlmodelc/weights/weight.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Decoder.mlmodelc/weights/weight.bin",
        expected_size: 23_604_992,
        expected_hash: 0xbb12_b352_be95_c813,
    },
    ModelFile {
        rel_path: "coreml/Encoder_v2.mlmodelc/analytics/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Encoder_v2.mlmodelc/analytics/coremldata.bin",
        expected_size: 243,
        expected_hash: 0x94f9_a8ab_bbd9_53ed,
    },
    ModelFile {
        rel_path: "coreml/Encoder_v2.mlmodelc/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Encoder_v2.mlmodelc/coremldata.bin",
        expected_size: 478,
        expected_hash: 0xce0d_2203_ec1a_be0f,
    },
    ModelFile {
        rel_path: "coreml/Encoder_v2.mlmodelc/model.mil",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Encoder_v2.mlmodelc/model.mil",
        expected_size: 1_002_653,
        expected_hash: 0xd3e1_b5e4_bfc3_a856,
    },
    ModelFile {
        rel_path: "coreml/Encoder_v2.mlmodelc/weights/weight.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Encoder_v2.mlmodelc/weights/weight.bin",
        expected_size: 594_211_328,
        expected_hash: 0x2126_0e3e_5f5a_1c6b,
    },
    ModelFile {
        rel_path: "coreml/JointDecision.mlmodelc/analytics/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/JointDecision.mlmodelc/analytics/coremldata.bin",
        expected_size: 243,
        expected_hash: 0x1e49_2793_ded8_4e93,
    },
    ModelFile {
        rel_path: "coreml/JointDecision.mlmodelc/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/JointDecision.mlmodelc/coremldata.bin",
        expected_size: 534,
        expected_hash: 0xe678_3b6a_11a3_0b20,
    },
    ModelFile {
        rel_path: "coreml/JointDecision.mlmodelc/metadata.json",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/JointDecision.mlmodelc/metadata.json",
        expected_size: 2_936,
        expected_hash: 0x8fb4_eb41_ce4a_cca9,
    },
    ModelFile {
        rel_path: "coreml/JointDecision.mlmodelc/model.mil",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/JointDecision.mlmodelc/model.mil",
        expected_size: 9_723,
        expected_hash: 0x6691_29f7_6dd4_e7c5,
    },
    ModelFile {
        rel_path: "coreml/JointDecision.mlmodelc/weights/weight.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/JointDecision.mlmodelc/weights/weight.bin",
        expected_size: 12_642_764,
        expected_hash: 0x823e_7ce6_7b69_035e,
    },
    ModelFile {
        rel_path: "coreml/Preprocessor.mlmodelc/analytics/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Preprocessor.mlmodelc/analytics/coremldata.bin",
        expected_size: 243,
        expected_hash: 0x88ac_15e9_4a10_4d1b,
    },
    ModelFile {
        rel_path: "coreml/Preprocessor.mlmodelc/coremldata.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Preprocessor.mlmodelc/coremldata.bin",
        expected_size: 486,
        expected_hash: 0xb033_4690_afd3_86a5,
    },
    ModelFile {
        rel_path: "coreml/Preprocessor.mlmodelc/metadata.json",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Preprocessor.mlmodelc/metadata.json",
        expected_size: 2_841,
        expected_hash: 0x9b49_43e6_eafc_7a1a,
    },
    ModelFile {
        rel_path: "coreml/Preprocessor.mlmodelc/model.mil",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Preprocessor.mlmodelc/model.mil",
        expected_size: 28_181,
        expected_hash: 0x50b8_d74d_7e9b_40e4,
    },
    ModelFile {
        rel_path: "coreml/Preprocessor.mlmodelc/weights/weight.bin",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/Preprocessor.mlmodelc/weights/weight.bin",
        expected_size: 491_072,
        expected_hash: 0x4073_accd_7d37_b82c,
    },
    ModelFile {
        rel_path: "coreml/parakeet_v3_vocab.json",
        url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/parakeet_v3_vocab.json",
        expected_size: 151_122,
        expected_hash: 0x2b2b_f8e6_57e9_4102,
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
    let mut file =
        File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
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
    let file_name = dest_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let temp_name = format!("{}.download.{}.tmp", file_name, pid);
    let temp_path = dest_path.with_file_name(temp_name);

    // Remove any leftover temporary file
    let _ = fs::remove_file(&temp_path);

    eprintln!("Downloading {} ...\n  URL: {}", file_name, url);

    let status = Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("--fail") // Fail on HTTP 4xx/5xx
        .arg("--progress-bar") // Clean progress display
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
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "curl download failed with exit code: {:?}",
            status.code()
        ));
    }

    // Verify downloaded file size
    let downloaded_size = match fs::metadata(&temp_path) {
        Ok(meta) => meta.len(),
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "Failed to read metadata of {}: {}",
                temp_path.display(),
                e
            ));
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
            return Err(format!(
                "Failed to compute hash for {}: {}",
                temp_path.display(),
                e
            ));
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

    eprintln!(
        "Downloaded and verified: {} (hash: {:016x})",
        file_name, downloaded_hash
    );
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
            download_file(model.url, &dest, model.expected_size, model.expected_hash)?;
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

    // The CoreML framework links Swift's runtime by @rpath; macOS ships it in
    // /usr/lib/swift.
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,/usr/lib/swift");
    // The statically linked FFmpeg contains C++ code that needs libc++.
    println!("cargo:rustc-link-lib=c++");

    if let Err(e) = check_and_download_group(&models_dir, COREML_MODELS, "coreml") {
        eprintln!("error: failed to prepare CoreML models: {}", e);
        std::process::exit(1);
    }
    eprintln!("All models verified in models/.\n");
}
