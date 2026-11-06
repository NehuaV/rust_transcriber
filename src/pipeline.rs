//! Transcribing a whole recording: cut it into windows, decode every window
//! several ways on worker threads, and stitch the results.
//!
//! Parakeet TDT v3 decodes some cuts of clear speech to nothing, a property of
//! the model (FluidAudio issue 909). The output also depends on small input
//! details, so a cut that blanks one way often works another way. Every window
//! is therefore decoded several ways ([`VARIANTS`]) and the result with the most
//! words is kept.
//!
//! This is deliberately free of assumptions about the audio: nothing decides
//! from loudness or silence what to decode or skip, and the choice between
//! variants compares only the model's own outputs. A blank variant has no words
//! and loses to one that has some. No word is ever edited; each window's result
//! is one complete model output.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use anyhow::{anyhow, Context, Result};

use crate::audio::SAMPLE_RATE;
use crate::coreml::{Engine, LengthPolicy, WINDOW_SAMPLES};
use crate::window::{self, Plan};
use crate::word::{self, Word};

/// Audio shared by neighbouring windows: 4 s. Every kept word had at least half
/// of this as context on both sides.
const OVERLAP_SAMPLES: usize = 4 * SAMPLE_RATE as usize;

/// Most windows decoded at once.
///
/// Each worker loads its own copy of the models, which are memory-mapped, so
/// this costs little RAM. Beyond about four the Neural Engine, which runs the
/// encoder one window at a time, is the limit.
const MAX_THREADS: usize = 4;

/// Variants used unless the caller asks for another number.
pub const DEFAULT_VARIANTS: u8 = 6;

/// One way of decoding a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// Seconds to move the window's cut. The audio decoded is still one window
    /// long, but starts this much earlier (negative) or later.
    shift_secs: i64,
    /// What the models are told about the audio's length.
    policy: LengthPolicy,
}

/// Shorthand for building [`VARIANTS`].
const fn variant(shift_secs: i64, policy: LengthPolicy) -> Variant {
    Variant { shift_secs, policy }
}

/// The ways a window can be decoded, best first. Using the first `n` of them
/// costs about `n` times the time of one.
pub const VARIANTS: [Variant; 12] = [
    variant(0, LengthPolicy::ACTUAL),
    variant(0, LengthPolicy::TRIMMED_TAIL),
    variant(-1, LengthPolicy::ACTUAL),
    variant(0, LengthPolicy::ENCODER_FULL_TRIMMED),
    variant(1, LengthPolicy::ACTUAL),
    variant(0, LengthPolicy::PREPROCESSOR_FULL_TRIMMED),
    variant(-2, LengthPolicy::ACTUAL),
    variant(2, LengthPolicy::ACTUAL),
    variant(-1, LengthPolicy::TRIMMED_TAIL),
    variant(1, LengthPolicy::TRIMMED_TAIL),
    variant(-3, LengthPolicy::ACTUAL),
    variant(3, LengthPolicy::ACTUAL),
];

/// How many windows are done, out of how many. Called after each window.
pub type Progress<'a> = &'a (dyn Fn(usize, usize) + Sync);

/// Transcribes 16 kHz mono `samples` with the CoreML models in `model_dir`,
/// decoding every window `variants` ways (clamped to `1..=VARIANTS.len()`).
///
/// # Errors
///
/// If the models cannot be loaded or a window fails to decode.
pub fn transcribe(
    samples: &[f32],
    model_dir: &Path,
    variants: usize,
    progress: Progress<'_>,
) -> Result<String> {
    let variants = variants.clamp(1, VARIANTS.len());
    let plan = Plan::new(samples.len(), WINDOW_SAMPLES, OVERLAP_SAMPLES);
    let windows = decode_windows(samples, model_dir, &plan, variants, progress)?;
    Ok(word::join(&window::merge(windows)))
}

/// Work every worker thread shares.
struct Shared<'a> {
    samples: &'a [f32],
    model_dir: &'a Path,
    plan: &'a Plan,
    variants: usize,
    progress: Progress<'a>,
    /// Index of the next window nobody has taken.
    next: AtomicUsize,
    /// Windows finished.
    done: AtomicUsize,
    /// Set when a worker fails, so the others stop early.
    failed: AtomicBool,
}

/// Decodes every window of `plan` on worker threads and returns the words each
/// owns, in window order.
fn decode_windows(
    samples: &[f32],
    model_dir: &Path,
    plan: &Plan,
    variants: usize,
    progress: Progress<'_>,
) -> Result<Vec<Vec<Word>>> {
    let shared = Shared {
        samples,
        model_dir,
        plan,
        variants,
        progress,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        failed: AtomicBool::new(false),
    };

    let finished = thread::scope(|scope| {
        let workers: Vec<_> = (0..thread_count(plan.len()))
            .map(|_| scope.spawn(|| work(&shared)))
            .collect();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| anyhow!("A decoding thread panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let mut windows = vec![Vec::new(); plan.len()];
    for (index, words) in finished.into_iter().flatten() {
        windows[index] = words;
    }
    Ok(windows)
}

/// Worker threads for `windows` windows: the cores available, up to
/// [`MAX_THREADS`], and never more than there are windows.
fn thread_count(windows: usize) -> usize {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    cores.min(MAX_THREADS).min(windows).max(1)
}

/// One worker: loads its own engine, then takes windows until none are left.
/// Returns each window it decoded with its index.
fn work(shared: &Shared<'_>) -> Result<Vec<(usize, Vec<Word>)>> {
    let mut decoded = Vec::new();
    let engine = match Engine::load(shared.model_dir) {
        Ok(engine) => engine,
        Err(error) => {
            shared.failed.store(true, Ordering::Relaxed);
            return Err(error);
        }
    };
    while !shared.failed.load(Ordering::Relaxed) {
        let index = shared.next.fetch_add(1, Ordering::Relaxed);
        if index >= shared.plan.len() {
            break;
        }
        match decode_window(&engine, shared.samples, shared.plan, index, shared.variants) {
            Ok(words) => decoded.push((index, words)),
            Err(error) => {
                shared.failed.store(true, Ordering::Relaxed);
                return Err(error);
            }
        }
        let done = shared.done.fetch_add(1, Ordering::Relaxed) + 1;
        (shared.progress)(done, shared.plan.len());
    }
    Ok(decoded)
}

/// Decodes window `index` the first `variants` ways of [`VARIANTS`] and returns
/// the words of the variant with the most.
fn decode_window(
    engine: &Engine,
    samples: &[f32],
    plan: &Plan,
    index: usize,
    variants: usize,
) -> Result<Vec<Word>> {
    let window = plan.spans()[index].clone();
    let len = window.len();
    let rate = i64::from(SAMPLE_RATE);

    let mut best: Option<Vec<Word>> = None;
    for variant in VARIANTS.iter().take(variants) {
        let start = (window.start as i64 + variant.shift_secs * rate).clamp(0, samples.len() as i64)
            as usize;
        let end = (start + len).min(samples.len());
        let same_cut = start == window.start && end == window.end;
        if end <= start || (best.is_some() && variant.shift_secs != 0 && same_cut) {
            // Nothing to decode, or a shift that clamped back to the first cut.
            continue;
        }

        let words = engine
            .transcribe(&samples[start..end], variant.policy)
            .with_context(|| {
                format!(
                    "CoreML inference failed (window {}/{})",
                    index + 1,
                    plan.len()
                )
            })?;
        let owned = plan.own_words(index, words, start);
        if best.as_ref().is_none_or(|best| owned.len() > best.len()) {
            best = Some(owned);
        }
    }
    Ok(best.unwrap_or_default())
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
