//! Words with timestamps, and the text operations on them.
//!
//! The model emits sub-word pieces; [`crate::coreml`] assembles them into
//! [`Word`]s. Everything here is plain text handling with no model or audio
//! involved.

/// A word with start and end times in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    /// The word as it appears in the transcript, with any punctuation that
    /// closes it.
    pub text: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

impl Word {
    /// Midpoint in seconds. Windows assign a word to one side of a cut by its
    /// midpoint.
    pub fn midpoint(&self) -> f32 {
        (self.start + self.end) * 0.5
    }
}

/// Whether `text` is a single punctuation mark that attaches to the word
/// before it.
fn is_trailing_punct(text: &str) -> bool {
    matches!(
        text.as_bytes(),
        [b'.' | b',' | b'!' | b'?' | b';' | b':' | b')']
    )
}

/// Glues trailing punctuation onto the previous word, keeping that word's
/// timing, so a period that lands just past a window cut stays with the word it
/// closes.
pub fn fold_trailing_punctuation(words: Vec<Word>) -> Vec<Word> {
    let mut folded: Vec<Word> = Vec::with_capacity(words.len());
    for word in words {
        match folded.last_mut() {
            Some(previous) if is_trailing_punct(&word.text) => previous.text.push_str(&word.text),
            _ => folded.push(word),
        }
    }
    folded
}

/// Whether two words are the same word: their letters and digits match,
/// ignoring case and punctuation. Two words with no letters or digits are never
/// the same.
pub fn same_word(a: &str, b: &str) -> bool {
    let key = |word: &str| -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let (a, b) = (key(a), key(b));
    !a.is_empty() && a == b
}

/// Joins words with single spaces, attaching trailing punctuation to the word
/// before it.
pub fn join(words: &[Word]) -> String {
    let mut text = String::new();
    for (index, word) in words.iter().enumerate() {
        if index > 0 && !is_trailing_punct(&word.text) {
            text.push(' ');
        }
        text.push_str(&word.text);
    }
    text
}

#[cfg(test)]
#[path = "word_tests.rs"]
mod tests;
