//! Cutting a recording into overlapping windows and stitching the words back
//! together.
//!
//! The CoreML encoder only takes 15 seconds of audio, so a recording is cut into
//! windows that overlap. Each cut sits in the middle of the overlap, and a word
//! belongs to the window that contains its midpoint, so every word is kept by
//! exactly one window and was heard with at least half the overlap of context on
//! both sides.

use std::ops::Range;

use crate::audio::SAMPLE_RATE;
use crate::word::{fold_trailing_punctuation, same_word, Word};

/// Seconds within which a repeated word across a window seam is the same word.
///
/// Neighbouring windows time the same word slightly differently. A real
/// repetition ("the the") is spoken later than this and is kept.
const SEAM_DUPLICATE_SECS: f32 = 0.25;

/// How a recording is cut into overlapping windows.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Sample range of every window, in order.
    spans: Vec<Range<usize>>,
    /// Samples shared by neighbouring windows.
    overlap: usize,
}

impl Plan {
    /// Plans windows of at most `window` samples over `total` samples, each
    /// overlapping the next by `overlap` samples. The last window always runs
    /// through the end of the recording. A recording that fits in one window
    /// gets exactly one.
    ///
    /// # Panics
    ///
    /// If `window` does not exceed `overlap`.
    pub fn new(total: usize, window: usize, overlap: usize) -> Self {
        assert!(window > overlap, "window must be longer than the overlap");
        let mut spans = Vec::new();
        if total > 0 && total <= window {
            spans.push(0..total);
        } else if total > 0 {
            let stride = window - overlap;
            let mut start = 0;
            loop {
                let end = (start + window).min(total);
                spans.push(start..end);
                if end == total {
                    break;
                }
                start += stride;
            }
        }
        Self { spans, overlap }
    }

    /// Number of windows.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Sample range of every window, in order.
    pub fn spans(&self) -> &[Range<usize>] {
        &self.spans
    }

    /// Seconds, measured from the start of the recording, whose words belong
    /// to window `index`.
    ///
    /// The cut sits in the middle of the overlap, so each window extends half
    /// the overlap past it. The first window keeps everything from the start,
    /// and the last keeps everything after its cut, including timestamps that
    /// land slightly past the final sample.
    pub fn keep_range(&self, index: usize) -> Range<f32> {
        let span = &self.spans[index];
        let rate = SAMPLE_RATE as f32;
        let half_overlap = (self.overlap / 2) as f32 / rate;
        let from = if index == 0 {
            0.0
        } else {
            span.start as f32 / rate + half_overlap
        };
        let to = if index + 1 == self.spans.len() {
            f32::MAX
        } else {
            span.end as f32 / rate - half_overlap
        };
        from..to
    }

    /// Keeps the words of window `index` that it owns and moves their
    /// timestamps to the whole recording.
    ///
    /// `words` were decoded from audio starting at sample `decoded_from` (the
    /// window's own start, or a shifted cut) and carry timestamps relative to
    /// that. Trailing punctuation is folded onto the word it closes first.
    pub fn own_words(&self, index: usize, words: Vec<Word>, decoded_from: usize) -> Vec<Word> {
        let keep = self.keep_range(index);
        let offset = decoded_from as f32 / SAMPLE_RATE as f32;
        fold_trailing_punctuation(words)
            .into_iter()
            .filter_map(|mut word| {
                let owned =
                    keep.contains(&(offset + word.midpoint())) && !word.text.trim().is_empty();
                owned.then(|| {
                    word.start += offset;
                    word.end += offset;
                    word
                })
            })
            .collect()
    }
}

/// Concatenates the words each window owns, in window order.
///
/// A word that sits at a seam can be kept by both neighbouring windows, because
/// they time it slightly differently. A word that repeats the previous window's
/// last word at almost the same time is that duplicate and is dropped. Repeats
/// inside a window are never touched.
pub fn merge(windows: Vec<Vec<Word>>) -> Vec<Word> {
    let mut merged: Vec<Word> = Vec::new();
    for window in windows {
        let seam = merged.len();
        for word in window {
            let duplicate = merged.len() == seam
                && merged.last().is_some_and(|previous| {
                    same_word(&previous.text, &word.text)
                        && (word.start - previous.start).abs() < SEAM_DUPLICATE_SECS
                });
            if !duplicate {
                merged.push(word);
            }
        }
    }
    merged
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
