# Parakeet Transcriber 🦜🎙️

A standalone, high-performance speech-to-text CLI tool and web application powered by **[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** and **[`parakeet-rs`](https://crates.io/crates/parakeet-rs)**.

Built with **pure Rust** and **ONNX Runtime (`ort`)**, it transcribes **any audio or video format** (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, WEBM, MP4, MOV, MKV, etc.) with automatic capitalization, punctuation, and sub-second processing.

---

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Compilation & Building](#-compilation--building)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Direct Binary Usage](#1-direct-binary-usage)
  - [Via `./start.sh`](#2-via-startsh)
  - [Via `cargo run`](#3-via-cargo-run)
  - [Additional Options & Flags](#4-additional-options--flags)
  - [CLI Reference](#5-cli-reference)
- [Model Architecture & Storage](#-model-architecture--storage)
- [Web Application UI](#-web-application-ui)
- [Project Layout](#-project-layout)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## ✨ Features

- 🦀 **Pure Rust Implementation**: Built using the `parakeet-rs` crate (`0.4.3`) and ONNX Runtime. Zero Python runtime or environment required for CLI transcription.
- ⚡ **High Throughput & Low Latency**: Transcribes at ~5x real-time speed on consumer hardware.
- 🎧 **Universal Format Support**: Accepts any container and codec (M4A, MP3, WAV, AAC, OGG, OPUS, FLAC, WEBM, MP4, MOV, MKV).
- 🔤 **Automatic Punctuation & Capitalization**: Built directly into the FastConformer-TDT architecture without needing a separate language model.
- 🌍 **Multilingual**: Supports 25 European languages with automatic language detection.
- 📊 **Rich Metrics & Output Modes**: Live terminal spinner, duration, processing speedup factor (RTFx), word count, clean text output (`-q`), and structured JSON output (`--json`).

---

## 🛠️ Prerequisites

Before compiling, ensure you have the following installed on your system:

### 1. Rust & Cargo
Rust `1.80+` (edition 2021):
```bash
# Verify Rust installation
rustc --version
cargo --version

# If not installed, install via rustup:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. FFmpeg & FFprobe
Used for decoding and preprocessing arbitrary media containers:
```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Verify installation
ffmpeg -version
ffprobe -version
```

---

## ⚙️ Compilation & Building

Clone the repository and compile the project using Cargo:

### Debug Build (Fast Compilation)
```bash
cargo build
```
The compiled binary will be placed at:
```
target/debug/transcriber
```

### Release Build (Optimized for Speed)
```bash
cargo build --release
```
The optimized release binary will be placed at:
```
target/release/transcriber
```

To copy the binary directly to the root of the project:
```bash
cp target/release/transcriber ./transcriber
```

---

## 📦 Installation

To make the `transcriber` command available globally anywhere in your terminal:

### Option A: Install via Cargo
```bash
cargo install --path .
```
This places the `transcriber` executable in `~/.cargo/bin/`. Ensure `~/.cargo/bin` is in your system's `PATH`:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Option B: Copy to `/usr/local/bin`
```bash
sudo cp target/release/transcriber /usr/local/bin/transcriber
```

Once installed, you can run `transcriber` from any folder:
```bash
transcriber audio.m4a -o transcript.txt
```

---

## 🚀 Usage

### 1. Direct Binary Usage

Transcribe an audio file and print the formatted transcript to the terminal:

```bash
./transcriber path/to/recording.m4a
```

Transcribe and save directly to an output text file:

```bash
./transcriber path/to/recording.m4a -o transcript.txt
```

---

### 2. Via `./start.sh`

The `./start.sh` launcher automatically delegates audio file arguments directly to the compiled Rust binary:

```bash
# Direct transcription to a file
./start.sh path/to/recording.m4a -o transcript.txt

# Output to terminal
./start.sh path/to/recording.mp3
```

---

### 3. Via `cargo run`

You can also run directly from source code without manually copying the binary:

```bash
cargo run --release -- path/to/recording.m4a -o transcript.txt
```

---

### 4. Additional Options & Flags

#### Output Full Metadata as JSON
Produces structured JSON including text, duration, processing time, real-time factor, and word count:

```bash
./transcriber recording.m4a --json
```

Example JSON output:
```json
{
  "text": "This is an example recording transcribed by Parakeet.",
  "duration": 6.10,
  "processing_time": 1.28,
  "rtfx": 4.8,
  "model": "nvidia/parakeet-tdt-0.6b-v3",
  "word_count": 8,
  "char_count": 52
}
```

Save JSON directly to a file:
```bash
./transcriber recording.m4a --json -o result.json
```

#### Quiet Mode (Clean Output for Piping)
Suppresses banners, metrics, and progress spinners, outputting only the transcript text. Ideal for shell piping or copying directly to clipboard:

```bash
# Copy transcript directly to macOS clipboard
./transcriber recording.m4a -q | pbcopy

# Pipe transcript to another command
./transcriber recording.m4a -q | grep -i "action item"
```

#### Custom Model Directory
Specify an alternate local directory containing ONNX model files:

```bash
./transcriber recording.m4a --model-dir /path/to/custom_model_dir -o output.txt
```

---

### 5. CLI Reference

```
Usage: transcriber [OPTIONS] <INPUT_FILE>

Arguments:
  <INPUT_FILE>
          Path to the audio or video file to transcribe

Options:
  -o, --output <OUTPUT_FILE>
          Optional path to save the transcript text (or JSON if --json is passed)

      --model-dir <MODEL_DIR>
          Optional custom path to directory containing ONNX weights
          (defaults to ~/.cache/parakeet-rs/parakeet-tdt-0.6b-v3/)

      --json
          Output full transcription result as JSON

  -q, --quiet
          Only output the transcribed text, suppressing banners and metrics

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

---

## 🧠 Model Architecture & Storage

### Model Files
The Rust CLI uses **`parakeet-rs`** with the ONNX export of **NVIDIA Parakeet TDT 0.6B v3**:
- **`encoder-model.int8.onnx`** (~620 MB): FastConformer acoustic encoder (quantized int8 for fast inference).
- **`decoder_joint-model.int8.onnx`** (~17 MB): TDT joint and token duration prediction network.
- **`vocab.txt`** (~92 KB): Tokenizer vocabulary mapping.

### Cache Location
Model files are stored locally in:
```
~/.cache/parakeet-rs/parakeet-tdt-0.6b-v3/
```
If the files are missing on first execution, the binary will automatically download them from Hugging Face (`istupakov/parakeet-tdt-0.6b-v3-onnx`).

---

## 🌐 Web Application UI

In addition to the Rust CLI tool, this repository includes an interactive local web application with drag-and-drop audio transcription:

### Launching the Web App
```bash
./start.sh
```
*(or `uv run transcriber`)*

This starts the local FastAPI server and automatically opens your browser at:
```
http://127.0.0.1:8000
```

### Web App Features
- **Drag-and-Drop Zone**: Drop any audio/video file directly into the browser.
- **Audio Player**: Preview playback with speed controls (`0.75x`, `1.0x`, `1.25x`, `1.5x`, `2.0x`).
- **Interactive Transcript**: Edit and proofread transcripts directly in the UI.
- **Search & Highlight**: In-transcript instant keyword search.
- **Exports**: One-click download as Plain Text (`.txt`), Markdown (`.md`), or JSON (`.json`).

---

## 📁 Project Layout

```
transcriber/
├── Cargo.toml                 # Rust package configuration & parakeet-rs dependency
├── Cargo.lock                 # Pinned Rust dependencies
├── rust/
│   └── src/
│       └── main.rs            # Pure Rust CLI implementation using parakeet-rs
├── transcriber                # Compiled standalone release binary
├── start.sh                   # Unified launcher script (CLI & Web UI)
├── pyproject.toml             # Python project dependencies (FastAPI, Transformers)
├── src/
│   └── transcriber/
│       ├── cli.py             # Python CLI & auto-browser launcher
│       ├── engine.py          # PyTorch MPS inference engine
│       ├── server.py          # FastAPI web backend
│       └── static/
│           └── index.html     # Dark-mode responsive drag-and-drop UI
└── README.md                  # Complete documentation
```

---

## ❓ Troubleshooting & FAQ

### 1. `ffprobe` or `ffmpeg` not found
**Cause**: FFmpeg is not installed or not in your system `PATH`.  
**Solution**: Install FFmpeg via Homebrew on macOS (`brew install ffmpeg`) or package manager on Linux (`sudo apt install ffmpeg`).

### 2. Can I transcribe video files?
**Yes**: You can pass `.mp4`, `.mov`, `.mkv`, or `.webm` files directly. The tool uses FFmpeg under the hood to extract the audio stream and convert it to 16kHz mono PCM before passing it to the neural network.

### 3. How do I transcribe multiple files?
You can use standard shell loops:
```bash
for file in audio_files/*.m4a; do
    ./transcriber "$file" -o "${file%.m4a}.txt"
done
```

### 4. How do I change the model cache directory?
Set the `PARAKEET_MODEL_DIR` environment variable:
```bash
export PARAKEET_MODEL_DIR="/path/to/my/models"
./transcriber audio.m4a
```
Or pass `--model-dir /path/to/my/models`.
