# Parakeet Transcriber 🦜🎙️

A completely **self-contained**, high-performance speech-to-text CLI tool written in **Rust**, powered by **[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)**, **[`parakeet-rs`](https://crates.io/crates/parakeet-rs)**, and **native statically linked FFmpeg** (`ffmpeg-next` / `ffmpeg-sys-next`).

Transcribe **any audio or video format** (M4A, MP3, WAV, FLAC, OGG, OPUS, AAC, WEBM, MP4, MOV, MKV, etc.) with automatic capitalization, punctuation, and sub-second processing.

---

## ✨ Features

- 🦀 **Self-Contained Executable with Native Static Linking**:
  - **Embedded FFmpeg C Libraries**: The FFmpeg decoding routines (`libavcodec.a`, `libavformat.a`, `libavutil.a`, `libswresample.a`) are statically linked directly into the machine code of the binary. No dynamic linking to FFmpeg, and no spawning external FFmpeg CLI subprocesses.
  - **Embedded Neural Model**: The INT8 quantized NVIDIA Parakeet TDT model weights (~640 MB) and tokenizer vocabulary are embedded directly into the binary via Rust `include_bytes!`. Completely operational offline with zero network calls at runtime.
  - **In-Memory Streaming Decoding & Resampling**: FFmpeg reads and decodes the container directly in memory and resamples audio to 16 kHz mono PCM on the fly.
- ⚡ **Blazing Fast**: Achieves **12x - 14x real-time** processing speed on Apple Silicon & modern x86_64 CPUs (e.g. 21s of audio transcribed in ~1.6s).
- 🎧 **Universal Format Support**: Accepts every container and codec supported by FFmpeg (M4A, MP3, WAV, AAC, OGG, OPUS WhatsApp voice notes, FLAC, WEBM, MP4, MOV, etc.).
- 🔤 **Automatic Punctuation & Capitalization**: Native FastConformer-TDT architecture outputs clean, punctuated sentences without needing external language model post-processors.
- 📊 **Rich Output Modes**:
  - Live animated terminal spinner & execution metrics.
  - Formatted terminal report with real-time factor (RTFx) and word counts.
  - Plain text file export (`-o transcript.txt`).
  - Clean text output for Unix pipes or clipboard (`-q`).
  - Structured JSON output with metadata (`--json`).

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Compilation & Building](#-compilation--building)
- [Usage](#-usage)
  - [1. Direct Binary Usage](#1-direct-binary-usage)
  - [2. Via `./start.sh`](#2-via-startsh)
  - [3. Via `cargo run`](#3-via-cargo-run)
  - [4. Saving Transcripts](#4-saving-transcripts)
  - [5. JSON Output Mode](#5-json-output-mode)
  - [6. Quiet Mode (Piping & Clipboard)](#6-quiet-mode-piping--clipboard)
- [CLI Reference](#-cli-reference)
- [Project Layout](#-project-layout)

---

## 🛠️ Prerequisites

To compile the application from source:

1. **Rust 1.80+** and **Cargo**:
   ```bash
   rustc --version
   cargo --version
   ```

2. **Static FFmpeg Development Libraries** (at build time only):
   - **macOS (Homebrew)**:
     ```bash
     brew install ffmpeg
     ```
     The static `.a` archives located in `/opt/homebrew/Cellar/ffmpeg/.../lib` are automatically detected and statically linked into the final binary.
   - **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
     ```

> [!NOTE]
> Once compiled, the end-user receives a standalone binary that needs **zero system commands** (`ffmpeg`, `ffprobe`, `python`, `curl`) and **no external model downloads**.

---

## ⚙️ Compilation & Building

The repository contains the ONNX INT8 model weights inside `models/`. Compiling with Cargo embeds them directly into the standalone binary:

```bash
cargo build --release
```

The optimized binary is created at `target/release/transcriber`.

To copy the binary directly to the root of the project:
```bash
cp target/release/transcriber ./transcriber
```

---

## 🚀 Usage

### 1. Direct Binary Usage

Transcribe an audio file and display formatted output in the terminal:
```bash
./transcriber audio/whatsapp_note_ale.opus
```

Example terminal output:
```
🦜 PARAKEET TRANSCRIBER [Embedded Model • Statically Linked FFmpeg • Zero System Dependencies]
🎵 Input: whatsapp_note_ale.opus (00:21, opus, 52.3 KB)
⚡ Engine: parakeet-rs | Speed: 13.1x real-time (in 1.62s) | Words: 62

────────────────────────────────────────────────────────────
TRANSCRIPTION:
────────────────────────────────────────────────────────────
Yeah no I I wanted to ask you like why you you needed that but I don't know what happened. I I was in uh in the car uh with the you know like the the voice things and I think he he didn't understand what I wanted to do. This is funny. Okay no uh I mean why you needed it
────────────────────────────────────────────────────────────
```

---

### 2. Via `./start.sh`

The `./start.sh` launcher forwards all arguments directly to `./transcriber` (and automatically compiles the release binary if it hasn't been built yet):

```bash
# Transcribe to a file
./start.sh path/to/recording.m4a -o transcript.txt

# Quiet mode
./start.sh audio/whatsapp_note_ale.opus -q
```

---

### 3. Via `cargo run`

Run directly through Cargo without copying the binary:
```bash
cargo run --release -- path/to/recording.m4a -o transcript.txt
```

---

### 4. Saving Transcripts

Use the `-o` or `--output` flag to write the transcript directly to a file:
```bash
./transcriber audio/whatsapp_note_ale.opus -o transcript.txt
```

---

### 5. JSON Output Mode

Pass `--json` to receive structured JSON metadata:
```bash
./transcriber audio/whatsapp_note_ale.opus --json
```

Output:
```json
{
  "text": "Yeah no I I wanted to ask you like why you you needed that but I don't know what happened. I I was in uh in the car uh with the you know like the the voice things and I think he he didn't understand what I wanted to do. This is funny. Okay no uh I mean why you needed it",
  "duration": 21.31,
  "processing_time": 1.595,
  "rtfx": 13.4,
  "model": "nvidia/parakeet-tdt-0.6b-v3 [embedded]",
  "word_count": 62,
  "char_count": 270
}
```

You can combine `--json` with `-o` to write formatted JSON directly to disk:
```bash
./transcriber audio/whatsapp_note_ale.opus --json -o output.json
```

---

### 6. Quiet Mode (Piping & Clipboard)

Pass `-q` or `--quiet` to output only the transcription text with zero banners, progress spinners, or metrics:

```bash
# Copy transcription directly to macOS clipboard
./transcriber audio/whatsapp_note_ale.opus -q | pbcopy

# Pipe into a search filter
./transcriber audio/whatsapp_note_ale.opus -q | grep -i "needed"
```

---

## 📖 CLI Reference

```
⚡ Completely self-contained speech-to-text CLI powered by embedded NVIDIA Parakeet TDT 0.6B v3 and statically linked FFmpeg

Usage: transcriber [OPTIONS] <INPUT_FILE>

Arguments:
  <INPUT_FILE>
          Path to the audio or video file to transcribe

Options:
  -o, --output <OUTPUT_FILE>
          Optional output file path to save the transcription

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

## 📁 Project Layout

```
transcriber/
├── .cargo/
│   └── config.toml            # Target linker configuration for static FFmpeg dependencies
├── Cargo.toml                 # Cargo manifest (ffmpeg-next, ffmpeg-sys-next static, parakeet-rs, clap)
├── Cargo.lock                 # Pinned dependencies
├── models/                    # Quantized ONNX weights embedded into binary at compile time
│   ├── encoder-model.int8.onnx (~622 MB)
│   ├── decoder_joint-model.int8.onnx (~17 MB)
│   └── vocab.txt (~92 KB)
├── rust/
│   └── src/
│       └── main.rs            # Statically linked FFmpeg decoding & Parakeet TDT inference
├── audio/
│   └── whatsapp_note_ale.opus # Sample Opus audio voice note
├── start.sh                   # Executable launcher script
├── transcriber                # Compiled release binary (~677 MB, self-contained)
└── README.md                  # Comprehensive documentation
```
