//! NVIDIA Parakeet TDT 0.6B v3 on Apple CoreML.
//!
//! Runs the FluidInference CoreML export of the model through the safe `coreml`
//! crate, on the Neural Engine, GPU and CPU as CoreML chooses. The export is
//! fixed-shape: the encoder always sees a 15 second window (240,000 samples), so
//! the caller cuts longer audio into windows (see [`crate::window`]).
//!
//! Decoding one window:
//!
//! ```text
//! audio [1, 240000] -> Preprocessor -> mel     [1, 128, 1501]
//!                   -> Encoder_v2   -> encoder [1, 1024, 188]
//! for each encoder frame: Decoder (LSTM state) + JointDecision -> token, duration
//! ```
//!
//! `Encoder_v2` is the int8 per-channel encoder; the original palettized
//! `Encoder` mangles words near window edges (FluidAudio issue 760).
//!
//! A window can be decoded more than one way ([`LengthPolicy`]). Every result is
//! the model's own output, exactly as returned.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use coreml::prelude::*;

use crate::word::Word;

/// Samples the fixed-shape encoder accepts per call (15 s at 16 kHz).
pub const WINDOW_SAMPLES: usize = 240_000;
/// Samples between mel frames.
const HOP_SAMPLES: usize = 160;
/// Encoder frames for a full window (the encoder subsamples mel frames by 8).
const ENCODER_FRAMES: usize = 188;
/// Encoder hidden size.
const ENCODER_DIM: usize = 1024;
/// Hidden size of the prediction network (an LSTM).
const DECODER_DIM: usize = 640;
/// Layers of the prediction network.
const LSTM_LAYERS: usize = 2;
/// Id of the blank token, which sits after the 8192 vocabulary entries.
const BLANK_ID: usize = 8192;
/// Most tokens emitted at one encoder frame before the decoder must advance.
const MAX_SYMBOLS_PER_FRAME: usize = 10;
/// Seconds per encoder frame: a 160-sample hop at 16 kHz, subsampled by 8.
const FRAME_SECS: f32 = 0.08;
/// Audio the trimmed-tail policy declares absent: 0.2 s at 16 kHz.
const TRIMMED_TAIL_SAMPLES: usize = 3_200;

/// How the models are told where the audio ends inside the padded 15 s input.
///
/// Parakeet TDT v3 has input cuts on which the joint network never beats blank,
/// and the output is on a knife edge with respect to the length inputs
/// (FluidAudio issue 909; the ONNX build and the fp16 MLX port blank on the same
/// cuts, so it is the model, not CoreML). Declaring the lengths differently
/// flips such a cut, which makes it possible to decode a window more than one
/// way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LengthPolicy {
    /// Declare the whole 15 s valid to the preprocessor.
    preprocessor_full: bool,
    /// Declare the whole mel length valid to the encoder.
    encoder_full: bool,
    /// Declare the audio 0.2 s shorter and silence that tail.
    trimmed_tail: bool,
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

    /// Samples the model is told the audio has, for `len` samples of real audio.
    fn declared_samples(self, len: usize) -> usize {
        if self.trimmed_tail && len > TRIMMED_TAIL_SAMPLES {
            len - TRIMMED_TAIL_SAMPLES
        } else {
            len
        }
    }
}

/// Number of encoder frames that cover `samples` of audio: mel frames are one
/// per [`HOP_SAMPLES`] and the encoder subsamples them by 8, capped at a full
/// window.
fn encoder_frames(samples: usize) -> usize {
    let mel_frames = 1 + samples / HOP_SAMPLES;
    (1 + (mel_frames - 1) / 8).min(ENCODER_FRAMES)
}

/// One token the joint network emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Token {
    /// Vocabulary id.
    id: usize,
    /// Encoder frame it was emitted at.
    frame: usize,
    /// Frames the joint network said it spans.
    duration: usize,
}

/// The vocabulary: the text of every token id.
#[derive(Debug, Clone, Default)]
struct Vocab(Vec<String>);

impl Vocab {
    /// Reads `parakeet_v3_vocab.json`.
    fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .with_context(|| format!("Failed to read vocabulary {}", path.display()))?;
        Self::from_json(&json)
    }

    /// Parses a JSON object mapping each id (as a string) to its text.
    ///
    /// # Errors
    ///
    /// If the JSON is not such an object, an id is not a number, or ids do not
    /// fill `0..len`.
    fn from_json(json: &str) -> Result<Self> {
        let map: HashMap<String, String> =
            serde_json::from_str(json).context("Vocabulary is not a JSON object of strings")?;
        let mut pieces = vec![String::new(); map.len()];
        for (id, piece) in map {
            let id: usize = id.parse().context("Vocabulary id is not a number")?;
            match pieces.get_mut(id) {
                Some(slot) => *slot = piece,
                None => bail!("Vocabulary id {id} is out of range"),
            }
        }
        Ok(Self(pieces))
    }

    /// Text of token `id`, if it exists.
    fn get(&self, id: usize) -> Option<&str> {
        self.0.get(id).map(String::as_str)
    }
}

/// Turns tokens into words with timestamps.
///
/// A word starts at a piece marked with `▁` (SentencePiece) or, in the
/// vocabulary JSON where `▁` was already replaced, with a leading space. Special
/// tokens such as `<|nospeech|>` are dropped. Digit pieces carry no marker, so a
/// run of digits that follows letters starts a new word ("at60" becomes
/// "at 60").
fn words_from_tokens(tokens: &[Token], vocab: &Vocab) -> Vec<Word> {
    /// Whether `word` ends in letters that make a following digit a new word:
    /// more than one letter, or the article "a".
    fn ends_in_letters(word: &Word) -> bool {
        let letters = word
            .text
            .chars()
            .rev()
            .take_while(|c| c.is_alphabetic())
            .count();
        letters > 1 || (letters == 1 && word.text.ends_with('a'))
    }

    let mut words: Vec<Word> = Vec::new();
    // Whether the last word can still take more pieces.
    let mut word_open = false;

    for token in tokens {
        let Some(piece) = vocab.get(token.id) else {
            continue;
        };
        if piece.starts_with('<') && piece.ends_with('>') && piece != "<unk>" {
            continue;
        }

        let starts_word = piece.starts_with(['▁', ' ']);
        let text = piece.trim_start_matches(['▁', ' ']);
        if text.is_empty() {
            word_open = false;
            continue;
        }

        let start = token.frame as f32 * FRAME_SECS;
        let end = (token.frame + token.duration.max(1)) as f32 * FRAME_SECS;

        let digits_after_letters = !starts_word
            && text.chars().all(|c| c.is_ascii_digit())
            && words.last().is_some_and(ends_in_letters);

        match words.last_mut() {
            Some(word) if !starts_word && !digits_after_letters && word_open => {
                word.text.push_str(text);
                word.end = end;
            }
            _ => {
                words.push(Word {
                    text: text.to_owned(),
                    start,
                    end,
                });
                word_open = true;
            }
        }
    }
    words
}

/// Loads a CoreML bundle on `units`.
fn load_model(path: &Path, units: ComputeUnits) -> Result<Model> {
    if !path.exists() {
        bail!("CoreML model not found: {}", path.display());
    }
    let configuration = ModelConfiguration::new().with_compute_units(units);
    Model::load_from_url(path, &configuration)
        .map_err(|e| anyhow!("Failed to load CoreML model {}: {e}", path.display()))
}

/// A new `f32` array with the given shape and contents.
fn f32_array(shape: &[usize], values: &[f32]) -> Result<MultiArray> {
    let mut array = MultiArray::new_f32(shape).map_err(|e| anyhow!("MultiArray alloc: {e}"))?;
    array
        .copy_from_slice(values)
        .map_err(|e| anyhow!("MultiArray fill: {e}"))?;
    Ok(array)
}

/// A new `i32` array with the given shape and contents.
fn i32_array(shape: &[usize], values: &[i32]) -> Result<MultiArray> {
    let mut array = MultiArray::new_i32(shape).map_err(|e| anyhow!("MultiArray alloc: {e}"))?;
    array
        .copy_from_slice(values)
        .map_err(|e| anyhow!("MultiArray fill: {e}"))?;
    Ok(array)
}

/// Reads any numeric array into a row-major `Vec<f32>`. The crate follows the
/// array's strides, so layouts padded by the Neural Engine read correctly.
fn to_f32(array: &MultiArrayRef) -> Result<Vec<f32>> {
    let values = match array.data_type() {
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
        other => bail!("Unsupported MLMultiArray data type {other:?}"),
    };
    values.map_err(|e| anyhow!("Failed to read MLMultiArray: {e}"))
}

/// The single number in a scalar array.
fn scalar(array: &MultiArrayView<'_>) -> Result<i32> {
    to_f32(array)?
        .first()
        .map(|&value| value as i32)
        .context("CoreML returned an empty scalar array")
}

/// An owned copy of an array, so it can be fed to the next model.
fn copy_array(array: &MultiArrayRef) -> Result<MultiArray> {
    array
        .copy_to_owned()
        .map_err(|e| anyhow!("Failed to copy MLMultiArray: {e}"))
}

/// Runs `model` on named inputs.
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

/// The output called `name`, as a read-only view.
fn output<'a>(outputs: &'a FeatureProvider, name: &str) -> Result<MultiArrayView<'a>> {
    outputs
        .get_multi_array(name)
        .with_context(|| format!("CoreML output '{name}' is missing"))
}

/// An owned copy of the output called `name`.
fn output_copy(outputs: &FeatureProvider, name: &str) -> Result<MultiArray> {
    let view = output(outputs, name)?;
    copy_array(&view)
}

/// What the prediction network produced for one input token.
struct DecoderStep {
    /// Its output, which the joint network combines with an encoder frame.
    output: MultiArray,
    /// LSTM hidden state to commit if the token is emitted.
    h: MultiArray,
    /// LSTM cell state to commit if the token is emitted.
    c: MultiArray,
}

/// The loaded CoreML models and the vocabulary for one worker.
///
/// A `coreml::Model` cannot be shared between threads, so each worker thread
/// loads its own engine. The weights are memory-mapped, so extra engines cost
/// little memory.
pub struct Engine {
    preprocessor: Model,
    encoder: Model,
    decoder: Model,
    joint: Model,
    vocab: Vocab,
}

impl Engine {
    /// Loads the models and vocabulary from `dir`.
    ///
    /// The preprocessor and encoder run wherever CoreML decides. The decoder
    /// and joint network are tiny and called thousands of times per file, so
    /// they are pinned to the CPU, where they skip the Neural Engine's dispatch
    /// latency, which dominated their runtime.
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            preprocessor: load_model(&dir.join("Preprocessor.mlmodelc"), ComputeUnits::All)?,
            encoder: load_model(&dir.join("Encoder_v2.mlmodelc"), ComputeUnits::All)?,
            decoder: load_model(&dir.join("Decoder.mlmodelc"), ComputeUnits::CpuOnly)?,
            joint: load_model(&dir.join("JointDecision.mlmodelc"), ComputeUnits::CpuOnly)?,
            vocab: Vocab::load(&dir.join("parakeet_v3_vocab.json"))?,
        })
    }

    /// Transcribes one window of 16 kHz mono audio, at most
    /// [`WINDOW_SAMPLES`] long. Word timestamps are in seconds from the start
    /// of `samples`.
    ///
    /// # Errors
    ///
    /// If the window is too long or a model fails.
    pub fn transcribe(&self, samples: &[f32], policy: LengthPolicy) -> Result<Vec<Word>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if samples.len() > WINDOW_SAMPLES {
            bail!(
                "Window of {} samples exceeds the {WINDOW_SAMPLES} sample CoreML limit",
                samples.len()
            );
        }
        let (frames, count) = self.encode(samples, policy)?;
        let tokens = self.decode(&frames, count)?;
        Ok(words_from_tokens(&tokens, &self.vocab))
    }

    /// Runs the preprocessor and encoder. Returns the encoder frames, laid out
    /// `[frame][ENCODER_DIM]` so each frame is contiguous, and how many are
    /// valid.
    fn encode(&self, samples: &[f32], policy: LengthPolicy) -> Result<(Vec<f32>, usize)> {
        let declared = policy.declared_samples(samples.len());
        let mut padded = vec![0f32; WINDOW_SAMPLES];
        padded[..declared].copy_from_slice(&samples[..declared]);
        let preprocessor_length = if policy.preprocessor_full {
            WINDOW_SAMPLES
        } else {
            declared
        };

        let mel = predict(
            &self.preprocessor,
            vec![
                ("audio_signal", f32_array(&[1, WINDOW_SAMPLES], &padded)?),
                (
                    "audio_length",
                    i32_array(&[1], &[preprocessor_length as i32])?,
                ),
            ],
        )?;
        let mel_length = if policy.encoder_full {
            i32_array(&[1], &[(1 + WINDOW_SAMPLES / HOP_SAMPLES) as i32])?
        } else {
            output_copy(&mel, "mel_length")?
        };
        let encoded = predict(
            &self.encoder,
            vec![
                ("mel", output_copy(&mel, "mel")?),
                ("mel_length", mel_length),
            ],
        )?;

        let encoder = output(&encoded, "encoder")?;
        let shape = encoder.shape();
        if shape != [1, ENCODER_DIM, ENCODER_FRAMES] {
            bail!(
                "Unexpected encoder output shape {shape:?}, expected [1, {ENCODER_DIM}, {ENCODER_FRAMES}]"
            );
        }
        let data = to_f32(&encoder)?;
        // The decoder stops at the real frames even when the encoder was told
        // the whole window is valid.
        let valid = encoder_frames(declared);

        // The model lays frames out as [dim][frame]; transpose to [frame][dim].
        let mut frames = vec![0f32; valid * ENCODER_DIM];
        for (frame, chunk) in frames.chunks_mut(ENCODER_DIM).enumerate() {
            for (dim, value) in chunk.iter_mut().enumerate() {
                *value = data[dim * ENCODER_FRAMES + frame];
            }
        }
        Ok((frames, valid))
    }

    /// Runs the prediction network on `last_token` and the LSTM state.
    fn run_decoder(
        &self,
        last_token: usize,
        h: &MultiArray,
        c: &MultiArray,
    ) -> Result<DecoderStep> {
        let out = predict(
            &self.decoder,
            vec![
                ("targets", i32_array(&[1, 1], &[last_token as i32])?),
                ("target_length", i32_array(&[1], &[1])?),
                ("h_in", copy_array(h)?),
                ("c_in", copy_array(c)?),
            ],
        )?;
        Ok(DecoderStep {
            output: output_copy(&out, "decoder")?,
            h: output_copy(&out, "h_out")?,
            c: output_copy(&out, "c_out")?,
        })
    }

    /// Runs the joint network on one encoder frame and the decoder output.
    /// Returns the predicted token id and how many frames it spans.
    fn run_joint(&self, frame: &[f32], decoder_output: &MultiArray) -> Result<(usize, usize)> {
        let out = predict(
            &self.joint,
            vec![
                ("encoder_step", f32_array(&[1, ENCODER_DIM, 1], frame)?),
                ("decoder_step", copy_array(decoder_output)?),
            ],
        )?;
        let id = scalar(&output(&out, "token_id")?)?.max(0) as usize;
        let duration = scalar(&output(&out, "duration")?)?.max(0) as usize;
        Ok((id, duration))
    }

    /// Greedy TDT decoding over the first `count` encoder frames.
    ///
    /// At each frame the joint network predicts a token and a duration. A
    /// non-blank token is emitted and advances the decoder; the duration says
    /// how many frames to skip. A zero duration stays on the frame, up to
    /// [`MAX_SYMBOLS_PER_FRAME`] tokens.
    fn decode(&self, frames: &[f32], count: usize) -> Result<Vec<Token>> {
        let zeros = vec![0f32; LSTM_LAYERS * DECODER_DIM];
        let mut h = f32_array(&[LSTM_LAYERS, 1, DECODER_DIM], &zeros)?;
        let mut c = f32_array(&[LSTM_LAYERS, 1, DECODER_DIM], &zeros)?;
        let mut last_token = BLANK_ID;
        // The decoder output for `last_token`, computed once and reused until
        // a token is emitted.
        let mut step: Option<DecoderStep> = None;

        let mut tokens = Vec::new();
        let mut frame = 0;
        let mut emitted_here = 0;

        while frame < count {
            let decoder = match step.take() {
                Some(cached) => cached,
                None => self.run_decoder(last_token, &h, &c)?,
            };
            let encoder_frame = &frames[frame * ENCODER_DIM..(frame + 1) * ENCODER_DIM];
            let (id, duration) = self.run_joint(encoder_frame, &decoder.output)?;

            if id == BLANK_ID {
                step = Some(decoder);
            } else {
                tokens.push(Token {
                    id,
                    frame,
                    duration,
                });
                last_token = id;
                h = decoder.h;
                c = decoder.c;
                emitted_here += 1;
            }

            if duration > 0 {
                frame += duration;
                emitted_here = 0;
            } else if id == BLANK_ID || emitted_here >= MAX_SYMBOLS_PER_FRAME {
                frame += 1;
                emitted_here = 0;
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
#[path = "coreml_tests.rs"]
mod tests;
