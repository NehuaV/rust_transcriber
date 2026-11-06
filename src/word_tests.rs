use super::*;

fn word(text: &str, start: f32, end: f32) -> Word {
    Word {
        text: text.to_string(),
        start,
        end,
    }
}

#[test]
fn midpoint_is_halfway_between_start_and_end() {
    assert_eq!(word("hi", 1.0, 2.0).midpoint(), 1.5);
}

#[test]
fn trailing_punctuation_keeps_the_word_timestamp() {
    let folded = fold_trailing_punctuation(vec![word("Hello", 1.0, 1.4), word(".", 1.4, 1.45)]);
    assert_eq!(folded, vec![word("Hello.", 1.0, 1.4)]);
}

#[test]
fn leading_punctuation_has_nothing_to_attach_to_and_stays() {
    let folded = fold_trailing_punctuation(vec![word(".", 0.0, 0.1), word("Hi", 0.2, 0.4)]);
    assert_eq!(folded.len(), 2);
}

#[test]
fn only_single_punctuation_marks_attach() {
    let folded = fold_trailing_punctuation(vec![word("a", 0.0, 0.1), word("...", 0.1, 0.2)]);
    assert_eq!(folded.len(), 2, "a run of dots is a word of its own");
}

#[test]
fn join_attaches_punctuation() {
    let words = [
        word("Hello", 0.0, 0.4),
        word(",", 0.4, 0.4),
        word("world", 0.5, 0.9),
        word(".", 0.9, 0.9),
    ];
    assert_eq!(join(&words), "Hello, world.");
}

#[test]
fn join_of_nothing_is_empty() {
    assert_eq!(join(&[]), "");
}

#[test]
fn same_word_ignores_case_and_punctuation() {
    assert!(same_word("river.", "River"));
    assert!(!same_word("river", "rivers"));
    assert!(!same_word(".", ","));
}
