use super::*;

fn vocab(pieces: &[&str]) -> Vocab {
    Vocab(pieces.iter().map(|piece| (*piece).to_owned()).collect())
}

fn token(id: usize, frame: usize, duration: usize) -> Token {
    Token {
        id,
        frame,
        duration,
    }
}

fn texts(words: &[Word]) -> Vec<&str> {
    words.iter().map(|word| word.text.as_str()).collect()
}

#[test]
fn vocab_json_is_indexed_by_id() {
    let vocab = Vocab::from_json(r#"{"1": "b", "0": "a", "2": "▁c"}"#).unwrap();
    assert_eq!(vocab.get(0), Some("a"));
    assert_eq!(vocab.get(2), Some("▁c"));
    assert_eq!(vocab.get(3), None);
}

#[test]
fn vocab_ids_must_fill_the_range() {
    assert!(Vocab::from_json(r#"{"5": "x"}"#).is_err());
    assert!(Vocab::from_json(r#"{"a": "x"}"#).is_err());
    assert!(Vocab::from_json("[1, 2]").is_err());
}

#[test]
fn pieces_join_into_words_with_timestamps() {
    let vocab = vocab(&["▁Hel", "lo", "▁world", "."]);
    let words = words_from_tokens(
        &[
            token(0, 0, 1),
            token(1, 1, 1),
            token(2, 3, 2),
            token(3, 5, 1),
        ],
        &vocab,
    );
    assert_eq!(texts(&words), ["Hello", "world."]);
    assert_eq!(words[0].start, 0.0);
    assert!((words[0].end - 2.0 * FRAME_SECS).abs() < 1e-6);
    assert!((words[1].start - 3.0 * FRAME_SECS).abs() < 1e-6);
    assert!((words[1].end - 6.0 * FRAME_SECS).abs() < 1e-6);
}

#[test]
fn special_tokens_are_skipped() {
    let vocab = vocab(&["<|nospeech|>", "▁hi", "<pad>"]);
    let words = words_from_tokens(&[token(0, 0, 1), token(1, 1, 1), token(2, 2, 1)], &vocab);
    assert_eq!(texts(&words), ["hi"]);
}

#[test]
fn the_unknown_token_is_kept() {
    let vocab = vocab(&["<unk>"]);
    let words = words_from_tokens(&[token(0, 0, 1)], &vocab);
    assert_eq!(texts(&words), ["<unk>"]);
}

#[test]
fn ids_outside_the_vocabulary_are_skipped() {
    let words = words_from_tokens(&[token(99, 0, 1)], &vocab(&["▁a"]));
    assert!(words.is_empty());
}

#[test]
fn digits_after_letters_start_a_new_word() {
    let vocab = vocab(&["▁like", "1", "0", "0"]);
    let words = words_from_tokens(
        &[
            token(0, 0, 1),
            token(1, 1, 1),
            token(2, 2, 1),
            token(3, 3, 1),
        ],
        &vocab,
    );
    assert_eq!(texts(&words), ["like", "100"]);
}

#[test]
fn space_prefixed_pieces_start_words() {
    // The CoreML vocabulary JSON marks word starts with a leading space.
    let vocab = vocab(&[" Damn", " no", ",", " I", "ama"]);
    let words = words_from_tokens(
        &[
            token(0, 0, 1),
            token(1, 1, 1),
            token(2, 2, 1),
            token(3, 3, 1),
            token(4, 4, 1),
        ],
        &vocab,
    );
    assert_eq!(texts(&words), ["Damn", "no,", "Iama"]);
}

#[test]
fn a_bare_space_piece_ends_the_current_word() {
    let vocab = vocab(&[" ", "ab", "cd"]);
    let words = words_from_tokens(&[token(1, 0, 1), token(0, 1, 1), token(2, 2, 1)], &vocab);
    assert_eq!(texts(&words), ["ab", "cd"]);
}

#[test]
fn zero_duration_tokens_still_have_an_end() {
    let words = words_from_tokens(&[token(0, 4, 0)], &vocab(&["▁a"]));
    assert!(words[0].end > words[0].start);
}

#[test]
fn encoder_frames_match_the_model() {
    assert_eq!(encoder_frames(240_000), 188);
    assert_eq!(encoder_frames(95_992), 75);
    assert_eq!(encoder_frames(239_984), 188);
    assert_eq!(
        encoder_frames(WINDOW_SAMPLES + 10_000),
        188,
        "capped at a window"
    );
}

#[test]
fn length_policies_declare_the_expected_length() {
    let len = 100_000;
    assert_eq!(LengthPolicy::ACTUAL.declared_samples(len), len);
    assert_eq!(
        LengthPolicy::TRIMMED_TAIL.declared_samples(len),
        len - TRIMMED_TAIL_SAMPLES
    );
    assert_eq!(
        LengthPolicy::ENCODER_FULL_TRIMMED.declared_samples(len),
        len - TRIMMED_TAIL_SAMPLES
    );
    // Audio shorter than the trim is left alone rather than underflowing.
    assert_eq!(LengthPolicy::TRIMMED_TAIL.declared_samples(100), 100);
}
