# Agent Guidelines

Welcome! When working in this repository as an AI agent or automated developer, follow these project conventions and instructions:

## Temporary Scripts & Testing

- **Scratch & Test Space**: You are encouraged to create temporary scripts, test harnesses, debugging utilities, and experimental code in the [`scripts/`](file:///Users/dobri/git/transcriber/scripts) folder (`/scripts`).
- **Usage**:
  - Use `/scripts` for scratch work such as validating audio decoding, prototyping new CLI options, running benchmarks, checking model tensors, or testing format edge cases.
  - Keep ad-hoc test scripts contained within the `/scripts` directory to maintain a clean root and project structure.
  - When scripts are no longer needed or testing is finalized, ensure the main codebase remains clean and self-contained.
- **Unit Tests**: Keep unit tests in separate files, never inline `mod tests { ... }` bodies:
  - [`src/transcribe_tests.rs`](file:///Users/dobri/git/transcriber/src/transcribe_tests.rs): windowing and stitching.
  - [`src/coreml_tests.rs`](file:///Users/dobri/git/transcriber/src/coreml_tests.rs): CoreML word assembly and gap detection.
  - Both are included from `src/transcribe.rs` via `#[path]`.
- **Which binary you test matters**: in the agent sandbox `CARGO_TARGET_DIR` points at a cache directory, so a binary in `./target/release/` can be stale. Run `"$CARGO_TARGET_DIR/release/transcribe"` after building, and check which file is newest.

## Project Architecture

- **macOS only, CoreML only**: This project targets macOS (Apple Silicon) and runs the model on CoreML. Do not add `cfg(target_os)` gating, Linux or Windows support, or another inference backend. An ONNX backend (`parakeet-rs`) and a streaming Nemotron binary (`stranscribe`) existed and were removed on purpose.
- **One binary**: `./transcribe`, source in [`src/transcribe.rs`](file:///Users/dobri/git/transcriber/src/transcribe.rs) and [`src/coreml.rs`](file:///Users/dobri/git/transcriber/src/coreml.rs), exported system-wide with [`./export.sh`](file:///Users/dobri/git/transcriber/export.sh) to `~/.local/bin`.
- **Model**: NVIDIA Parakeet TDT 0.6B v3 CoreML bundles from `models/coreml/`, run through the safe `coreml` crate. The TDT decode loop is our own Rust code. Uses `Encoder_v2` (int8 per-channel); the original palettized `Encoder` mangles words near window edges.
- **Windows can blank (a model property)**: Parakeet TDT v3 decodes some 15 second cuts of clear speech to nothing (FluidAudio issue 909). The encoder input is fixed at 15 s, so audio is windowed (15 s, 4 s overlap, stitched on word midpoints by `own_words`). Never use a window's output without the gap check (1.5 s of speech-level audio with no words). The recovery order in `decode_window` is: retry with perturbed `LengthPolicy` values (confidence gated), then re-decode with the cut shifted by -1, +1, -2, +2, -3, +3 seconds (actual length and every policy for each). Without checks a test file lost about 16% of its words.
- **Do not alter model output**: the checks may only add text where the model produced none. Never edit, drop or reorder words the model emitted (apart from overlap stitching). Report windows that stay incomplete (`RecoveryStats`) instead of hiding them.
- **No multithreading**: Parallel chunk workers were tried (on the earlier ONNX path) and removed: about 1.2x speedup for about double the memory. CoreML already schedules work across the Neural Engine, GPU and CPU.
- **Model Storage & Loading**:
  - Weights live in `models/coreml/`.
  - [`scripts/build.rs`](file:///Users/dobri/git/transcriber/scripts/build.rs) checks (size and hash) and downloads missing models during `cargo build` or via `cargo run --bin download-models`. It also links libc++ (the static FFmpeg has C++ code) and adds the `/usr/lib/swift` rpath that the `coreml` crate's Swift bridge needs.
  - The binary loads models from `models/` on disk without looking at `~/.cache` or making network requests at runtime.
- **Toolchain**: building requires Xcode with the Swift toolchain, because the `coreml` crate compiles a Swift bridge.
- **Static FFmpeg Linking**: The binary statically links FFmpeg via `ffmpeg-sys-next`. Do not add runtime dependencies on external system CLI binaries.
