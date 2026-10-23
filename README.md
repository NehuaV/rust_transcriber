# Transcriber

Offline speech-to-text CLI utilities written in Rust, built on NVIDIA Parakeet and Nemotron ASR models via `parakeet-rs` and statically linked FFmpeg.

The project compiles two standalone executables:

| Binary | Source | Model | Mode | Description |
| :--- | :--- | :--- | :--- | :--- |
| `transcribe` | `src/transcribe.rs` | NVIDIA Parakeet TDT 0.6B v3 | Batch | Non-streaming speech recognition for English audio |
| `stranscribe` | `src/stranscribe.rs` | NVIDIA Nemotron Multilingual 3.5 0.6B | Streaming | Cache-aware streaming recognition with support for 40+ languages |

---

## Architecture

- **Static FFmpeg linking**: Decoders and resamplers are statically compiled into each binary via `ffmpeg-sys-next`. No system `ffmpeg` binary or dynamic library installation is required at runtime.
- **Local model evaluation**: ONNX models run locally via ONNX Runtime without network access or disk writes to `~/.cache`.
- **Automatic model download**: A build script (`scripts/build.rs`) downloads required model weights on first build if they are not already present in `models/`.
- **Constant memory streaming**: `stranscribe` uses a producer-consumer decoding thread pipeline with a bounded channel, allowing long audio streams to be processed with $O(1)$ memory consumption.

---

## Prerequisites

Building from source requires:

1. **Rust 1.80+** and **Cargo**:
   ```bash
   rustc --version
   cargo --version
   ```

2. **FFmpeg development libraries** (compile-time only):
   - macOS:
     ```bash
     brew install ffmpeg
     ```
   - Linux (Debian/Ubuntu):
     ```bash
     sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
     ```

---

## Building and Installation

### 1. Build release binaries

Cargo executes `scripts/build.rs` to verify or download model files to `models/`, then compiles the binaries:

```bash
cargo build --release
```

Compiled executables are located at:
- `target/release/transcribe`
- `target/release/stranscribe`

To check or download models without building the CLI binaries:
```bash
cargo run --bin download-models
```

### 2. Export system-wide

Run `export.sh` to install symlinks to `~/.local/bin` (or pass a custom installation directory):

```bash
./export.sh
```

---

## Usage

### `transcribe` (Batch Recognition)

Transcribes full audio files using the Parakeet TDT 0.6B model.

```bash
# Basic transcription
transcribe audio.mp3

# Save transcript to file
transcribe audio.opus -o transcript.txt

# Quiet mode (stdout contains only transcribed text)
transcribe audio.m4a -q
transcribe audio.wav -q | pbcopy
```

#### Options

```
Usage: transcribe [OPTIONS] <INPUT_FILE>

Arguments:
  <INPUT_FILE>                 Path to the audio or video file to transcribe

Options:
  -o, --output <OUTPUT_FILE>   Optional output file path to save the transcription
  -q, --quiet                  Only output the transcribed text, suppressing metrics
  -h, --help                   Print help
  -V, --version                Print version
```

---

### `stranscribe` (Streaming Recognition)

Performs streaming transcription with immediate token output to stdout using the Nemotron Multilingual 3.5 model.

```bash
# Stream transcription (English default)
stranscribe audio.opus

# Specify target language
stranscribe audio.m4a -l es
stranscribe audio.mp3 -l de

# Automatic language detection
stranscribe audio.wav -l auto

# Save transcript while streaming live output
stranscribe meeting.m4a -o meeting.txt

# Quiet mode (stream text only, omit headers and speed metrics)
stranscribe voice_note.opus -q
```

#### Options

```
Usage: stranscribe [OPTIONS] <INPUT_FILE>

Arguments:
  <INPUT_FILE>                 Path to the audio or video file to transcribe

Options:
  -o, --output <OUTPUT_FILE>   Optional output file path to save the transcription
  -l, --language <LANGUAGE>    Target language code or locale [default: en]
  -q, --quiet                  Only output the transcribed text, suppressing metrics
  -h, --help                   Print help
  -V, --version                Print version
```

---

## Supported Languages (`stranscribe`)

The `stranscribe` binary supports 40+ language prompts via the `-l, --language` flag:

| Code | Language | Code | Language | Code | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `auto` | Auto-detect | `en` / `en-US` | English | `es` / `es-ES` | Spanish |
| `de` | German | `fr` | French | `it` | Italian |
| `pt` | Portuguese | `ru` | Russian | `zh-CN` | Mandarin Chinese |
| `ja` | Japanese | `ko` | Korean | `ar` | Arabic |
| `hi` | Hindi | `nl` | Dutch | `pl` | Polish |
| `tr` | Turkish | `uk` | Ukrainian | `sv` | Swedish |
| `da` | Danish | `fi` | Finnish | `no` | Norwegian |
| `cs` | Czech | `el` | Greek | `ro` | Romanian |
| `hu` | Hungarian | `sk` | Slovak | `hr` | Croatian |
| `bg` | Bulgarian | `lt` | Lithuanian | `et` | Estonian |
| `lv` | Latvian | `sl` | Slovenian | `th` | Thai |
| `vi` | Vietnamese | `id` | Indonesian | `ms` | Malay |

---

## Repository Layout

```
.
├── Cargo.toml               # Cargo package configuration
├── README.md                # Project documentation
├── AGENTS.md                # Agent guidelines and project structure notes
├── export.sh                # Installation script for ~/.local/bin
├── start.sh                 # Local launcher script
├── audio/                   # Sample audio files for testing
├── models/
│   ├── basic/               # Parakeet TDT 0.6B ONNX assets
│   │   ├── vocab.txt
│   │   ├── decoder_joint-model.int8.onnx
│   │   └── encoder-model.int8.onnx
│   └── streaming/           # Nemotron Multilingual 3.5 ONNX assets
│       ├── config.json
│       ├── tokenizer.model
│       ├── decoder_joint.onnx
│       ├── encoder.onnx
│       └── encoder.onnx.data
├── src/
│   ├── transcribe.rs        # Batch transcription CLI implementation
│   └── stranscribe.rs       # Streaming transcription CLI implementation
└── scripts/
    ├── build.rs             # Cargo build script for checking/downloading models
    └── download_models.rs   # Standalone runner for model downloading
```
