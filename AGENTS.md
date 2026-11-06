# Agent Guidelines

Welcome! When working in this repository as an AI agent or automated developer, follow these project conventions and instructions:

## Temporary Scripts & Testing

- **Scratch & Test Space**: You are encouraged to create temporary scripts, test harnesses, debugging utilities, and experimental code in the [`scripts/`](file:///Users/dobri/git/transcriber/scripts) folder (`/scripts`).
- **Usage**:
  - Use `/scripts` for scratch work such as validating audio decoding, prototyping new CLI options, running benchmarks, checking ONNX model tensors, or testing format edge cases.
  - Keep ad-hoc test scripts contained within the `/scripts` directory to maintain a clean root and project structure.
  - When scripts are no longer needed or testing is finalized, ensure the main codebase remains clean and self-contained.
- **Unit Tests**: Keep unit tests in a separate file, [`src/transcribe_tests.rs`](file:///Users/dobri/git/transcriber/src/transcribe_tests.rs), included from `src/transcribe.rs` via `#[path]`. Do not inline `mod tests { ... }` bodies in `transcribe.rs`.

## Project Architecture

- **Rust Standalone Binary**: The repository compiles one standalone CLI binary:
  - `./transcribe`: Source in [`src/transcribe.rs`](file:///Users/dobri/git/transcriber/src/transcribe.rs) using the NVIDIA Parakeet TDT 0.6B v3 model from `models/basic/`.
  - The binary can be exported system-wide using [`./export.sh`](file:///Users/dobri/git/transcriber/export.sh) to `~/.local/bin`.
  - The streaming model (Nemotron) and its `stranscribe` binary were removed. Chunked batch transcription gave better quality, so do not reintroduce a streaming path without a clear reason.
- **Long Audio Chunking**: The Parakeet TDT encoder only supports about 5000 encoder steps (about 400 seconds) per call because of its fixed relative-position table. `transcribe` splits longer audio into 5-minute chunks with 15 seconds of overlap and stitches them on word midpoints. Keep `CHUNK_SAMPLES` below `MAX_SAMPLES` in `src/transcribe.rs` when changing chunk sizes.
- **Model Storage & Loading**:
  - Model weights live in the `models/basic/` folder.
  - The build script in [`scripts/build.rs`](file:///Users/dobri/git/transcriber/scripts/build.rs) automatically checks and downloads missing models once during `cargo build` or via `cargo run --bin download-models`.
  - The binary loads models directly from `models/` on disk without looking at `~/.cache` or making any network requests at runtime.
- **Static FFmpeg Linking**: The binary statically links FFmpeg via `ffmpeg-sys-next`. Do not add runtime dependencies on external system CLI binaries.
