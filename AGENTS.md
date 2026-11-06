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
  - [`src/coreml_tests.rs`](file:///Users/dobri/git/transcriber/src/coreml_tests.rs): CoreML word assembly.
  - Both are included from `src/transcribe.rs` via `#[path]`.
- **Which binary you test matters**: in the agent sandbox `CARGO_TARGET_DIR` points at a cache directory, so a binary in `./target/release/` can be stale. Run `"$CARGO_TARGET_DIR/release/transcribe"` after building, and check which file is newest.

## Project Architecture

- **macOS only, CoreML only**: This project targets macOS (Apple Silicon) and runs the model on CoreML. Do not add `cfg(target_os)` gating, Linux or Windows support, or another inference backend. An ONNX backend (`parakeet-rs`) and a streaming Nemotron binary (`stranscribe`) existed and were removed on purpose.
- **One binary**: `./transcribe`, source in [`src/transcribe.rs`](file:///Users/dobri/git/transcriber/src/transcribe.rs) and [`src/coreml.rs`](file:///Users/dobri/git/transcriber/src/coreml.rs), exported system-wide with [`./export.sh`](file:///Users/dobri/git/transcriber/export.sh) to `~/.local/bin`.
- **Model**: NVIDIA Parakeet TDT 0.6B v3 CoreML bundles from `models/coreml/`, run through the safe `coreml` crate. The TDT decode loop is our own Rust code. Uses `Encoder_v2` (int8 per-channel); the original palettized `Encoder` mangles words near window edges.
- **Windowing and variants**: the encoder input is fixed at 15 s, so audio is cut into 15 s windows with 4 s overlap and stitched on word midpoints (`own_words`, `merge_windows`). Every window is decoded several ways (`VARIANTS`: different cut shifts and `LengthPolicy` length declarations, `--variants N`, default 6) and `decode_window` keeps the variant with the most words. Nothing is skipped: do not add heuristics that decide what to decode or keep from the audio (energy or silence thresholds), because the project owner rejected them.
- **Blank windows (a model property)**: Parakeet TDT v3 decodes some 15 s cuts of clear speech to nothing (FluidAudio issue 909; the ONNX build does it too). Decoding once loses about a quarter of the words on the 19 minute test file; the variants recover nearly all of it. Past about 6 variants the "most words" choice also starts picking invented phrases (for example "the American American American as Laredo"), so the default is 6. Windows that blank under every variant are still lost. Picking by token confidence instead of word count did not help.
- **Do not alter model output**: never edit, drop or reorder words the model emitted. Each window's result is one complete model output; the only post-processing is overlap stitching.
- **Speed**: windows are decoded by up to `MAX_WINDOW_THREADS` (4) worker threads, each loading its own `CoreMlEngine` (`coreml::Model` is not `Sync`). The models are memory-mapped, so RAM stays small, and the output must be identical for every thread count. The decoder and joint models are pinned to `ComputeUnits::CpuOnly` because Neural Engine dispatch latency dominated their runtime. More than 4 threads does not help because the Neural Engine serialises encoder runs. The joint network's cost is per-call overhead, so its compute unit does not matter.
- **Seam stitching**: `merge_windows` merges a word that two neighbouring windows both kept (same word, timed within 0.25 s). Do not widen that window much: a real repetition such as "the the" must survive.
- **Model Storage & Loading**:
  - Weights live in `models/coreml/`.
  - [`scripts/build.rs`](file:///Users/dobri/git/transcriber/scripts/build.rs) checks (size and hash) and downloads missing models during `cargo build` or via `cargo run --bin download-models`. It also links libc++ (the static FFmpeg has C++ code) and adds the `/usr/lib/swift` rpath that the `coreml` crate's Swift bridge needs.
  - The binary loads models from `models/` on disk without looking at `~/.cache` or making network requests at runtime.
- **Toolchain**: building requires Xcode with the Swift toolchain, because the `coreml` crate compiles a Swift bridge.
- **Static FFmpeg Linking**: The binary statically links FFmpeg via `ffmpeg-sys-next`. Do not add runtime dependencies on external system CLI binaries.
