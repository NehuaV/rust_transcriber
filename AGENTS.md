# Agent Guidelines

Welcome! When working in this repository as an AI agent or automated developer, follow these project conventions and instructions:

## Temporary Scripts & Testing

- **Scratch & Test Space**: You are encouraged to create temporary scripts, test harnesses, debugging utilities, and experimental code in the [`scripts/`](file:///Users/dobri/git/transcriber/scripts) folder (`/scripts`).
- **Usage**:
  - Use `/scripts` for scratch work such as validating audio decoding, prototyping new CLI options, running benchmarks, checking ONNX model tensors, or testing format edge cases.
  - Keep ad-hoc test scripts contained within the `/scripts` directory to maintain a clean root and project structure.
  - When scripts are no longer needed or testing is finalized, ensure the main codebase remains clean and self-contained.

## Project Architecture

- **Rust Standalone Binaries**: The repository compiles two separate standalone CLI binaries:
  - `./transcribe`: Source in [`src/transcribe.rs`](file:///Users/dobri/git/transcriber/src/transcribe.rs) using the NVIDIA Parakeet TDT 0.6B v3 model from `models/basic/`.
  - `./stranscribe`: Source in [`src/stranscribe.rs`](file:///Users/dobri/git/transcriber/src/stranscribe.rs) using the NVIDIA Nemotron Multilingual 3.5 0.6B streaming model from `models/streaming/`.
  - Both binaries can be exported system-wide using [`./export.sh`](file:///Users/dobri/git/transcriber/export.sh) to `~/.local/bin`.
- **Model Storage & Loading**:
  - Model weights live in the `models/` folder (`models/basic/` and `models/streaming/`).
  - The build script in [`scripts/build.rs`](file:///Users/dobri/git/transcriber/scripts/build.rs) automatically checks and downloads missing models once during `cargo build` or via `cargo run --bin download-models`.
  - Binaries load models directly from `models/` on disk without looking at `~/.cache` or making any network requests at runtime.
- **Static FFmpeg Linking**: Both binaries statically link FFmpeg via `ffmpeg-sys-next`. Do not add runtime dependencies on external system CLI binaries.
