use super::*;

#[test]
fn the_first_variant_is_the_plain_decode() {
    assert_eq!(VARIANTS[0], variant(0, LengthPolicy::ACTUAL));
}

#[test]
fn variants_are_all_different() {
    for (i, a) in VARIANTS.iter().enumerate() {
        assert!(!VARIANTS[i + 1..].contains(a), "duplicate variant {a:?}");
    }
}

#[test]
fn the_default_is_one_of_the_variants() {
    assert!((1..=VARIANTS.len()).contains(&usize::from(DEFAULT_VARIANTS)));
}

#[test]
fn shifts_stay_within_the_overlap_context() {
    // A shifted cut moves a few seconds at most, so it still decodes most of the
    // window's own audio and the overlap supplies the context it lost.
    let overlap_secs = (OVERLAP_SAMPLES as i64) / i64::from(SAMPLE_RATE);
    for variant in VARIANTS {
        assert!(variant.shift_secs.abs() < overlap_secs);
    }
}

#[test]
fn the_overlap_is_shorter_than_a_window() {
    const { assert!(OVERLAP_SAMPLES < WINDOW_SAMPLES) };
}

#[test]
fn thread_count_is_bounded() {
    assert_eq!(thread_count(0), 1);
    assert_eq!(thread_count(1), 1);
    assert!((1..=MAX_THREADS).contains(&thread_count(100)));
}
