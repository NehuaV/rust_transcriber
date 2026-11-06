use crate::coreml::{encoder_frames, parse_vocab, tokens_to_words, FRAME_SECS};
use crate::has_speech_gap;

fn vocab(pieces: &[&str]) -> Vec<String> {
    pieces.iter().map(|s| s.to_string()).collect()
}

#[test]
fn vocab_json_is_indexed_by_id() {
    let v = parse_vocab(r#"{"1": "b", "0": "a", "2": "▁c"}"#).unwrap();
    assert_eq!(v, vec!["a", "b", "▁c"]);
    assert!(parse_vocab(r#"{"5": "x"}"#).is_err());
}

#[test]
fn pieces_join_into_words_with_timestamps() {
    let v = vocab(&["▁Hel", "lo", "▁world", "."]);
    let words = tokens_to_words(&[(0, 0, 1), (1, 1, 1), (2, 3, 2), (3, 5, 1)], &v);
    let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, vec!["Hello", "world."]);
    assert_eq!(words[0].start, 0.0);
    assert!((words[0].end - 2.0 * FRAME_SECS).abs() < 1e-6);
    assert!((words[1].start - 3.0 * FRAME_SECS).abs() < 1e-6);
    assert!((words[1].end - 6.0 * FRAME_SECS).abs() < 1e-6);
}

#[test]
fn special_tokens_are_skipped() {
    let v = vocab(&["<|nospeech|>", "▁hi", "<pad>"]);
    let words = tokens_to_words(&[(0, 0, 1), (1, 1, 1), (2, 2, 1)], &v);
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].text, "hi");
}

#[test]
fn digits_after_letters_start_a_new_word() {
    let v = vocab(&["▁like", "1", "0", "0"]);
    let words = tokens_to_words(&[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)], &v);
    let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, vec!["like", "100"]);
}

#[test]
fn zero_duration_tokens_still_have_an_end() {
    let v = vocab(&["▁a"]);
    let words = tokens_to_words(&[(0, 4, 0)], &v);
    assert!(words[0].end > words[0].start);
}

#[test]
fn space_prefixed_pieces_start_words() {
    // The CoreML vocabulary JSON marks word starts with a leading space.
    let v = vocab(&[" Damn", " no", ",", " I", "ama"]);
    let words = tokens_to_words(&[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1)], &v);
    let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, vec!["Damn", "no,", "Iama"]);
}

#[test]
fn bare_space_piece_ends_the_current_word() {
    let v = vocab(&[" ", "ab", "cd"]);
    let words = tokens_to_words(&[(1, 0, 1), (0, 1, 1), (2, 2, 1)], &v);
    let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, vec!["ab", "cd"]);
}

fn tone(seconds: f32, amplitude: f32) -> Vec<f32> {
    let n = (seconds * 16_000.0) as usize;
    (0..n)
        .map(|i| amplitude * (i as f32 * 0.05).sin())
        .collect()
}

#[test]
fn speech_with_no_words_is_a_gap() {
    let window = tone(10.0, 0.1);
    assert!(has_speech_gap(&window, 0.0, 10.0, &[]));
}

#[test]
fn speech_with_words_throughout_is_not_a_gap() {
    let window = tone(10.0, 0.1);
    let words: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 + 0.25).collect();
    assert!(!has_speech_gap(&window, 0.0, 10.0, &words));
}

#[test]
fn silence_with_no_words_is_not_a_gap() {
    let window = vec![0.0f32; 160_000];
    assert!(!has_speech_gap(&window, 0.0, 10.0, &[]));
}

#[test]
fn a_long_skipped_stretch_inside_speech_is_a_gap() {
    let window = tone(15.0, 0.1);
    // Words at the start and end, nothing between 3 s and 9 s.
    let mut words: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 + 0.25).collect();
    words.extend((18..30).map(|i| i as f32 * 0.5 + 0.25));
    assert!(has_speech_gap(&window, 0.0, 15.0, &words));
}

#[test]
fn a_short_pause_is_not_a_gap() {
    let window = tone(10.0, 0.1);
    // Nothing between 4 s and 5 s: only 1 s.
    let mut words: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 + 0.25).collect();
    words.extend((10..20).map(|i| i as f32 * 0.5 + 0.25));
    assert!(!has_speech_gap(&window, 0.0, 10.0, &words));
}

#[test]
fn only_the_kept_range_is_checked() {
    let window = tone(15.0, 0.1);
    // Words only inside 6..10; the rest of the window is not ours to judge.
    let words: Vec<f32> = (12..20).map(|i| i as f32 * 0.5 + 0.25).collect();
    assert!(!has_speech_gap(&window, 6.0, 10.0, &words));
}

#[test]
fn encoder_frames_match_the_model() {
    assert_eq!(encoder_frames(240_000), 188);
    assert_eq!(encoder_frames(95_992), 75);
    assert_eq!(encoder_frames(239_984), 188);
}
