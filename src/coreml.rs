//! CoreML backend for NVIDIA Parakeet TDT 0.6B v3 (macOS only).
//!
//! Built on the safe `coreml` crate. Runs the FluidInference CoreML export of the model on the Apple Neural
//! Engine, GPU and CPU. The export is fixed-shape: the encoder always sees a
//! 15 second window (240,000 samples), so longer audio is windowed by the
//! caller. Pipeline for one window:
//!
//! ```text
//! audio [1, 240000] -> Preprocessor -> mel [1, 128, 1501]
//!                   -> Encoder_v2   -> encoder [1, 1024, 188]
//! per encoder frame: Decoder (LSTM state) + JointDecision -> token, duration
//! ```
//!
//! `Encoder_v2` is the int8 per-channel encoder. The original palettized
//! `Encoder` mangles words near window edges (FluidAudio issue 760).
//!
//! A window can be decoded more than one way (`LengthPolicy`); every result is
//! the model's own output, exactly as returned.

use anyhow::{anyhow, bail, Context, Result};
use coreml::prelude::*;
use std::path::Path;

/// A word (or token) with start and end times in seconds.
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Samples the fixed-shape encoder accepts per call (15 seconds at 16 kHz).
pub const WINDOW_SAMPLES: usize = 240_000;
/// Encoder frames for a full window (8x subsampling of the mel frames).
const ENCODER_FRAMES: usize = 188;
/// Encoder hidden size.
const ENCODER_DIM: usize = 1024;
/// Prediction network (LSTM) hidden size and layer count.
const DECODER_DIM: usize = 640;
const LSTM_LAYERS: usize = 2;
/// Blank token id; it sits after the 8192 vocabulary entries.
pub const BLANK_ID: usize = 8192;
/// Most tokens emitted at a single encoder frame before forcing an advance.
const MAX_SYMBOLS_PER_FRAME: usize = 10;
/// Seconds per encoder frame: 160-sample hop at 16 kHz, 8x subsampling.
pub const FRAME_SECS: f32 = 0.08;

/// One decoded token: vocabulary id, encoder frame it was emitted at, and the
/// number of frames the joint network said it spans.
pub type RawToken = (usize, usize, usize);

fn load_model(path: &Path) -> Result<Model> {
    load_model_with(path, ComputeUnits::All)
}

fn load_model_with(path: &Path, units: ComputeUnits) -> Result<Model> {
    if !path.exists() {
        bail!("CoreML model not found: {}", path.display());
    }
    let configuration = ModelConfiguration::new().with_compute_units(units);
    Model::load_from_url(path, &configuration)
        .map_err(|e| anyhow!("Failed to load CoreML model {}: {}", path.display(), e))
}

fn f32_array(shape: &[usize], values: &[f32]) -> Result<MultiArray> {
    let mut array = MultiArray::new_f32(shape).map_err(|e| anyhow!("MultiArray alloc: {e}"))?;
    array
        .copy_from_slice(values)
        .map_err(|e| anyhow!("MultiArray fill: {e}"))?;
    Ok(array)
}

fn i32_array(shape: &[usize], values: &[i32]) -> Result<MultiArray> {
    let mut array = MultiArray::new_i32(shape).map_err(|e| anyhow!("MultiArray alloc: {e}"))?;
    array
        .copy_from_slice(values)
        .map_err(|e| anyhow!("MultiArray fill: {e}"))?;
    Ok(array)
}

/// Read any numeric multi-array into a row-major `Vec<f32>`. The crate follows
/// the array's strides, so padded Neural Engine layouts read correctly.
fn to_f32(array: &MultiArrayRef) -> Result<Vec<f32>> {
    let read = match array.data_type() {
        DataType::Float32 => array.to_vec::<f32>(),
        DataType::Float16 => array
            .to_vec::<half::f16>()
            .map(|v| v.into_iter().map(f32::from).collect()),
        DataType::Float64 => array
            .to_vec::<f64>()
            .map(|v| v.into_iter().map(|x| x as f32).collect()),
        DataType::Int32 => array
            .to_vec::<i32>()
            .map(|v| v.into_iter().map(|x| x as f32).collect()),
        other => bail!("Unsupported MLMultiArray data type {:?}", other),
    };
    read.map_err(|e| anyhow!("Failed to read MLMultiArray: {e}"))
}

fn scalar(array: &MultiArrayView<'_>) -> Result<i32> {
    to_f32(array)?
        .first()
        .map(|&v| v as i32)
        .context("CoreML returned an empty scalar array")
}

/// Run a model on named inputs.
fn predict(model: &Model, inputs: Vec<(&str, MultiArray)>) -> Result<FeatureProvider> {
    let mut provider = FeatureProvider::new();
    for (name, array) in inputs {
        provider
            .insert_multi_array(name, array)
            .map_err(|e| anyhow!("Failed to set CoreML input '{name}': {e}"))?;
    }
    model
        .predict(&provider)
        .map_err(|e| anyhow!("CoreML prediction failed: {e}"))
}

/// Copy a named output into an owned array so it can feed the next model.
fn output_copy(out: &FeatureProvider, name: &str) -> Result<MultiArray> {
    out.get_multi_array(name)
        .with_context(|| format!("CoreML output '{name}' is missing"))?
        .copy_to_owned()
        .map_err(|e| anyhow!("Failed to copy CoreML output '{name}': {e}"))
}

fn output_view<'a>(out: &'a FeatureProvider, name: &str) -> Result<MultiArrayView<'a>> {
    out.get_multi_array(name)
        .with_context(|| format!("CoreML output '{name}' is missing"))
}

/// Encoder frames covering `samples`: `1 + (mel_frames - 1) / 8` with a
/// 160-sample hop, capped at a full window.
pub fn encoder_frames(samples: usize) -> usize {
    let mel_frames = 1 + samples / 160;
    (1 + (mel_frames - 1) / 8).min(ENCODER_FRAMES)
}

/// Audio the `trimmed_tail` policy declares absent (0.2 s at 16 kHz).
pub const TRIMMED_TAIL_SAMPLES: usize = 3_200;

/// How the models are told where the audio ends inside the padded 15 s input.
///
/// Parakeet TDT v3 has input cuts on which the joint never beats blank, and
/// the output is on a knife edge with respect to the length inputs (FluidAudio
/// issue 909; the ONNX build and the fp16 MLX port blank on the same cuts, so
/// it is the model, not CoreML). Declaring the lengths differently flips such a
/// cut, which makes it possible to decode a window more than one way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LengthPolicy {
    /// Declare the whole 15 s valid to the preprocessor.
    pub preprocessor_full: bool,
    /// Declare the whole mel length valid to the encoder.
    pub encoder_full: bool,
    /// Declare the audio 0.2 s shorter and silence that tail.
    pub trimmed_tail: bool,
}

impl LengthPolicy {
    /// Declare the true length.
    pub const ACTUAL: Self = Self::new(false, false, false);
    /// Declare the audio 0.2 s shorter.
    pub const TRIMMED_TAIL: Self = Self::new(false, false, true);
    /// Trimmed tail, with the whole mel length declared valid to the encoder.
    pub const ENCODER_FULL_TRIMMED: Self = Self::new(false, true, true);
    /// Trimmed tail, with the whole window declared valid to the preprocessor.
    pub const PREPROCESSOR_FULL_TRIMMED: Self = Self::new(true, false, true);

    const fn new(preprocessor_full: bool, encoder_full: bool, trimmed_tail: bool) -> Self {
        Self {
            preprocessor_full,
            encoder_full,
            trimmed_tail,
        }
    }
}

/// Samples the model is told the audio has under `policy`.
pub fn declared_samples(len: usize, policy: LengthPolicy) -> usize {
    if policy.trimmed_tail && len > TRIMMED_TAIL_SAMPLES {
        len - TRIMMED_TAIL_SAMPLES
    } else {
        len
    }
}

/// Loaded CoreML models plus the vocabulary.
pub struct CoreMlEngine {
    preprocessor: Model,
    encoder: Model,
    decoder: Model,
    joint: Model,
    vocab: Vec<String>,
}

impl CoreMlEngine {
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            preprocessor: load_model(&dir.join("Preprocessor.mlmodelc"))?,
            encoder: load_model(&dir.join("Encoder_v2.mlmodelc"))?,
            // The decoder and joint network are tiny and called thousands of
            // times per file. On the CPU they skip Neural Engine dispatch
            // latency, which dominated their runtime.
            decoder: load_model_with(&dir.join("Decoder.mlmodelc"), ComputeUnits::CpuOnly)?,
            joint: load_model_with(&dir.join("JointDecision.mlmodelc"), ComputeUnits::CpuOnly)?,
            vocab: load_vocab(&dir.join("parakeet_v3_vocab.json"))?,
        })
    }

    /// Transcribe at most one window of 16 kHz mono audio. Words carry
    /// timestamps in seconds relative to the start of `samples`.
    pub fn transcribe_window(
        &self,
        samples: &[f32],
        policy: LengthPolicy,
    ) -> Result<Vec<TimedToken>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if samples.len() > WINDOW_SAMPLES {
            bail!(
                "Window of {} samples exceeds the {} sample CoreML limit",
                samples.len(),
                WINDOW_SAMPLES
            );
        }

        let (encoder_out, frames) = self.encode(samples, policy)?;
        let tokens = self.decode(&encoder_out, frames)?;
        Ok(tokens_to_words(&tokens, &self.vocab))
    }

    /// Audio to encoder frames laid out as `[frame][ENCODER_DIM]`.
    fn encode(&self, samples: &[f32], policy: LengthPolicy) -> Result<(Vec<f32>, usize)> {
        let declared = declared_samples(samples.len(), policy);
        let mut padded = vec![0f32; WINDOW_SAMPLES];
        padded[..declared].copy_from_slice(&samples[..declared]);
        let audio = f32_array(&[1, WINDOW_SAMPLES], &padded)?;
        let pre_len = if policy.preprocessor_full {
            WINDOW_SAMPLES
        } else {
            declared
        };
        let length = i32_array(&[1], &[pre_len as i32])?;

        let mel_out = predict(
            &self.preprocessor,
            vec![("audio_signal", audio), ("audio_length", length)],
        )?;
        let mel_length = if policy.encoder_full {
            i32_array(&[1], &[(1 + WINDOW_SAMPLES / 160) as i32])?
        } else {
            output_copy(&mel_out, "mel_length")?
        };
        let enc_out = predict(
            &self.encoder,
            vec![
                ("mel", output_copy(&mel_out, "mel")?),
                ("mel_length", mel_length),
            ],
        )?;

        let encoder = output_view(&enc_out, "encoder")?;
        let shape = encoder.shape();
        if shape != [1, ENCODER_DIM, ENCODER_FRAMES] {
            bail!(
                "Unexpected encoder output shape {:?}, expected [1, {}, {}]",
                shape,
                ENCODER_DIM,
                ENCODER_FRAMES
            );
        }
        let data = to_f32(&encoder)?;
        // The decoder stops at the real frames even when the encoder was told
        // the whole window is valid.
        let valid = encoder_frames(declared);

        // [dim][frame] -> [frame][dim] so each frame is contiguous.
        let mut frames = vec![0f32; valid * ENCODER_DIM];
        for t in 0..valid {
            for d in 0..ENCODER_DIM {
                frames[t * ENCODER_DIM + d] = data[d * ENCODER_FRAMES + t];
            }
        }
        Ok((frames, valid))
    }

    /// Greedy TDT decoding over the encoder frames.
    fn decode(&self, frames: &[f32], count: usize) -> Result<Vec<RawToken>> {
        let zeros = vec![0f32; LSTM_LAYERS * DECODER_DIM];
        let mut h = f32_array(&[LSTM_LAYERS, 1, DECODER_DIM], &zeros)?;
        let mut c = f32_array(&[LSTM_LAYERS, 1, DECODER_DIM], &zeros)?;
        let mut last_token = BLANK_ID;
        // Decoder output for `last_token` plus the state it would commit.
        let mut pending: Option<(MultiArray, MultiArray, MultiArray)> = None;

        let mut tokens = Vec::new();
        let mut t = 0;
        let mut emitted = 0;

        while t < count {
            if pending.is_none() {
                let out = predict(
                    &self.decoder,
                    vec![
                        ("targets", i32_array(&[1, 1], &[last_token as i32])?),
                        ("target_length", i32_array(&[1], &[1])?),
                        ("h_in", h.copy_to_owned().map_err(|e| anyhow!("{e}"))?),
                        ("c_in", c.copy_to_owned().map_err(|e| anyhow!("{e}"))?),
                    ],
                )?;
                pending = Some((
                    output_copy(&out, "decoder")?,
                    output_copy(&out, "h_out")?,
                    output_copy(&out, "c_out")?,
                ));
            }
            let (decoder_step, next_h, next_c) = pending.as_ref().expect("decoder output cached");
            let joint = predict(
                &self.joint,
                vec![
                    (
                        "encoder_step",
                        f32_array(
                            &[1, ENCODER_DIM, 1],
                            &frames[t * ENCODER_DIM..(t + 1) * ENCODER_DIM],
                        )?,
                    ),
                    (
                        "decoder_step",
                        decoder_step.copy_to_owned().map_err(|e| anyhow!("{e}"))?,
                    ),
                ],
            )?;
            let token = scalar(&output_view(&joint, "token_id")?)?.max(0) as usize;
            let duration = scalar(&output_view(&joint, "duration")?)?.max(0) as usize;

            if token != BLANK_ID {
                tokens.push((token, t, duration));
                last_token = token;
                h = next_h.copy_to_owned().map_err(|e| anyhow!("{e}"))?;
                c = next_c.copy_to_owned().map_err(|e| anyhow!("{e}"))?;
                pending = None;
                emitted += 1;
            }

            if duration > 0 {
                t += duration;
                emitted = 0;
            } else if token == BLANK_ID || emitted >= MAX_SYMBOLS_PER_FRAME {
                t += 1;
                emitted = 0;
            }
        }
        Ok(tokens)
    }
}

/// Load `parakeet_v3_vocab.json`, a `{"id": "piece"}` map.
fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read vocabulary {}", path.display()))?;
    parse_vocab(&text)
}

pub fn parse_vocab(json: &str) -> Result<Vec<String>> {
    let map: std::collections::HashMap<String, String> =
        serde_json::from_str(json).context("Vocabulary is not a JSON object of strings")?;
    let mut vocab = vec![String::new(); map.len()];
    for (id, piece) in map {
        let id: usize = id.parse().context("Vocabulary id is not a number")?;
        if id >= vocab.len() {
            bail!("Vocabulary id {} is out of range", id);
        }
        vocab[id] = piece;
    }
    Ok(vocab)
}

/// Turn decoded tokens into words with timestamps.
///
/// A word start is marked by `▁` (SentencePiece) or, in the CoreML vocabulary
/// JSON where `▁` was already replaced, by a leading space. Special tokens such as
/// `<|nospeech|>` are dropped. Digit pieces carry no `▁`, so a digit run that
/// follows letters gets a space ("at60" becomes "at 60").
pub fn tokens_to_words(tokens: &[RawToken], vocab: &[String]) -> Vec<TimedToken> {
    let mut words: Vec<TimedToken> = Vec::new();
    let mut current_word_open = false;

    for &(id, frame, duration) in tokens {
        let Some(piece) = vocab.get(id) else { continue };
        if piece.starts_with('<') && piece.ends_with('>') && piece != "<unk>" {
            continue;
        }

        let starts_word = piece.starts_with(['▁', ' ']);
        let text = piece.trim_start_matches(['▁', ' ']);
        if text.is_empty() {
            current_word_open = false;
            continue;
        }

        let start = frame as f32 * FRAME_SECS;
        let end = (frame + duration.max(1)) as f32 * FRAME_SECS;

        let digit_after_letters = !starts_word
            && text.chars().all(|c| c.is_ascii_digit())
            && words.last().is_some_and(|w| {
                let letters = w
                    .text
                    .chars()
                    .rev()
                    .take_while(|c| c.is_alphabetic())
                    .count();
                let article_a = letters == 1 && w.text.ends_with('a');
                letters > 1 || article_a
            });

        if (starts_word || digit_after_letters || words.is_empty()) || !current_word_open {
            words.push(TimedToken {
                text: text.to_string(),
                start,
                end,
            });
            current_word_open = true;
        } else if let Some(word) = words.last_mut() {
            word.text.push_str(text);
            word.end = end;
        }
    }
    words
}
