# Transcriber

Offline speech-to-text CLI written in Rust, built on the NVIDIA Parakeet TDT 0.6B v3 model via `parakeet-rs` and statically linked FFmpeg.

| Binary | Source | Model | Description |
| :--- | :--- | :--- | :--- |
| `transcribe` | `src/transcribe.rs` | NVIDIA Parakeet TDT 0.6B v3 | Batch speech recognition with automatic chunking for long audio |

---

## Architecture

- **Static FFmpeg linking**: Decoders and resamplers are statically compiled into the binary via `ffmpeg-sys-next`. No system `ffmpeg` binary or dynamic library installation is required at runtime.
- **Local model evaluation**: ONNX models run locally via ONNX Runtime without network access or disk writes to `~/.cache`.
- **Automatic model download**: A build script (`scripts/build.rs`) downloads required model weights on first build if they are not already present in `models/`.
- **Long audio chunking**: The Parakeet TDT encoder cannot attend over more than about 5000 encoder steps (about 400 seconds) per call. Longer audio is split into overlapping windows and stitched back together.

### `transcribe` pipeline

```mermaid
flowchart TD
    CLI["CLI args<br/>INPUT_FILE, -o, -q"] --> Validate{"Input exists?"}
    Validate -- no --> Err1["Error: input does not exist"]
    Validate -- yes --> Decode

    subgraph Decode["1. decode_audio (static FFmpeg)"]
        direction TB
        Open["Open container<br/>M4A, MP3, WAV, OPUS, MP4, ..."] --> Pick["Pick best audio stream"]
        Pick --> Dec["Decode packets to frames"]
        Dec --> Resample["Resample to 16 kHz mono f32"]
        Resample --> PCM[("pcm_16k: Vec&lt;f32&gt;")]
    end

    Decode --> Locate["2. resolve_model_dir<br/>env override, ./models/basic, exe dir,<br/>~/.local/share/transcriber, manifest dir"]
    Locate --> Load["3. Load ONNX Runtime + ParakeetTDT<br/>encoder-model.int8.onnx<br/>decoder_joint-model.int8.onnx<br/>vocab.txt"]
    Load --> Route{"samples <= MAX_SAMPLES<br/>6,400,000 (400 s)?"}

    Route -- yes --> Single["Single pass<br/>transcribe_samples"]
    Route -- no --> Chunking

    subgraph Chunking["4. transcribe_pcm: chunked path"]
        direction TB
        Spans["chunk_spans<br/>5 min windows, 15 s overlap"] --> Loop["For each chunk:<br/>transcribe_samples with word timestamps"]
        Loop --> Fold["fold_trailing_punctuation<br/>attach . , ! ? to previous word"]
        Fold --> Keep["absolute_keep_window<br/>keep words whose midpoint is<br/>inside this chunk's half of the overlap"]
        Keep --> Loop
        Keep --> Join["join_words<br/>stitch kept words into one transcript"]
    end

    Single --> Text["Transcript text"]
    Join --> Text
    Text --> Out{"Quiet mode?"}
    Out -- yes --> Plain["stdout: text only"]
    Out -- no --> Report["stdout: header, speed (x real-time), words, text"]
    Plain --> Save{"-o given?"}
    Report --> Save
    Save -- yes --> File["Write transcript file"]
    Save -- no --> Done(["Done"])
    File --> Done
```

#### Stages

| Stage | Function | What it does |
| :--- | :--- | :--- |
| 1. Decode | `decode_audio` | FFmpeg demuxes, decodes and resamples any supported container to 16 kHz mono `f32` PCM held in memory |
| 2. Locate model | `resolve_model_dir` | Finds `models/basic` without touching `~/.cache` or the network |
| 3. Load model | `ParakeetTDT::from_pretrained` | Builds ONNX Runtime sessions for the encoder and decoder/joint graphs |
| 4. Transcribe | `transcribe_pcm` | One pass for short audio, overlapping chunks for long audio |
| 5. Output | `main` | Prints text (and metrics unless `-q`), optionally writes `-o` file |

#### Chunking details

The exported encoder uses a fixed relative-position table of 9999 entries, centered at 5000. Attention adds a slice of that table onto scores that are `2 * steps - 1` wide, so the add only broadcasts while `steps <= 5000`. At 16 kHz with a 160-sample hop and 8x subsampling, 6,400,000 samples is exactly 5000 steps; one hop more fails.

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `MAX_SAMPLES` | 6,400,000 (400 s) | Longest input sent in a single pass |
| `CHUNK_SAMPLES` | 4,800,000 (5 min) | Window length, safely under the limit |
| `OVERLAP_SAMPLES` | 240,000 (15 s) | Shared audio between neighboring windows |

```mermaid
flowchart LR
    subgraph Timeline["Audio timeline"]
        direction LR
        C1["Chunk 1<br/>0:00 - 5:00"] --- C2["Chunk 2<br/>4:45 - 9:45"] --- C3["Chunk 3<br/>9:30 - ..."]
    end
    C1 -. "cut at overlap midpoint (4:52.5)" .-> C2
    C2 -. "cut at overlap midpoint (9:37.5)" .-> C3
```

Each cut sits in the middle of the overlap. A word is kept by the chunk that contains its midpoint, so every word is emitted exactly once and was always heard with at least 7.5 seconds of context on both sides. The first chunk keeps everything from the start and the last chunk keeps everything to the end.

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

### 1. Build release binary

Cargo executes `scripts/build.rs` to verify or download model files to `models/`, then compiles the binary:

```bash
cargo build --release
```

The compiled executable is located at `target/release/transcribe`.

To check or download models without building the CLI binary:

```bash
cargo run --bin download-models
```

### 2. Export system-wide

Run `export.sh` to install the binary to `~/.local/bin` (or pass a custom installation directory):

```bash
./export.sh
```

### 3. Run tests

Unit tests live in `src/transcribe_tests.rs`:

```bash
cargo test --bin transcribe
```

---

## Usage

```bash
# Basic transcription
transcribe audio.mp3

# Save transcript to file
transcribe audio.opus -o transcript.txt

# Quiet mode (stdout contains only transcribed text)
transcribe audio.m4a -q
transcribe audio.wav -q | pbcopy
```

Long recordings (for example a 48 minute lecture) work the same way; chunking is automatic.

### Options

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
│   └── basic/               # Parakeet TDT 0.6B ONNX assets
│       ├── vocab.txt
│       ├── decoder_joint-model.int8.onnx
│       └── encoder-model.int8.onnx
├── src/
│   ├── transcribe.rs        # CLI, decoding, chunked transcription pipeline
│   └── transcribe_tests.rs  # Unit tests for chunking and stitching
└── scripts/
    ├── build.rs             # Cargo build script for checking/downloading models
    └── download_models.rs   # Standalone runner for model downloading
```
