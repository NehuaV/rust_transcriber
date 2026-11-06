//! Finding the CoreML model files on disk.
//!
//! Weights are loaded from local disk only. Nothing is read from `~/.cache` and
//! nothing is downloaded at runtime; `scripts/build.rs` fetches the files when
//! the project is built.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Folder under `models/` that holds the CoreML bundles.
const SUBDIR: &str = "coreml";
/// A file that exists only in a complete model folder.
const MARKER: &str = "parakeet_v3_vocab.json";
/// Environment variable that points at the model folder.
const DIR_ENV: &str = "PARAKEET_COREML_DIR";
/// Environment variable that points at a `models/` parent folder.
const PARENT_ENV: &str = "MODELS_DIR";

/// Returns the folder holding the CoreML models.
///
/// # Errors
///
/// If none of the places searched by [`candidates`] has the models.
pub fn coreml_dir() -> Result<PathBuf> {
    find(candidates()).context(
        "Parakeet CoreML models not found in 'models/coreml'.\n\
         Please run scripts/build.rs or execute:\n\
         cargo run --bin download-models",
    )
}

/// Every place the models may be, most specific first:
///
/// 1. the folder named by `PARAKEET_COREML_DIR` (or `MODELS_DIR`), or its
///    `coreml` subfolder,
/// 2. `models/coreml` under the current directory,
/// 3. next to the executable, one level up, and in `../share/transcriber`
///    (symlinks resolved, so an installed link finds the repository),
/// 4. `~/.local/share/transcriber/models/coreml`,
/// 5. the project folder this binary was built from.
fn candidates() -> Vec<PathBuf> {
    let relative = Path::new("models").join(SUBDIR);
    let mut places = Vec::new();

    if let Some(dir) = env::var_os(DIR_ENV).or_else(|| env::var_os(PARENT_ENV)) {
        let dir = PathBuf::from(dir);
        places.push(dir.clone());
        places.push(dir.join(SUBDIR));
    }

    places.push(relative.clone());

    if let Ok(exe) = env::current_exe() {
        let exe = fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(exe_dir) = exe.parent() {
            places.push(exe_dir.join(&relative));
            places.push(exe_dir.join("..").join(&relative));
            places.push(exe_dir.join("../share/transcriber").join(&relative));
        }
    }

    if let Some(home) = env::var_os("HOME") {
        places.push(
            Path::new(&home)
                .join(".local/share/transcriber")
                .join(&relative),
        );
    }

    places.push(Path::new(env!("CARGO_MANIFEST_DIR")).join(&relative));
    places
}

/// The first of `places` that holds a complete set of models.
fn find(places: Vec<PathBuf>) -> Option<PathBuf> {
    places.into_iter().find(|place| place.join(MARKER).exists())
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
