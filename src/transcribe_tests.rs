use super::*;

#[test]
fn durations_are_minutes_and_seconds() {
    assert_eq!(format_duration(0.0), "00:00");
    assert_eq!(format_duration(59.6), "01:00");
    assert_eq!(format_duration(1182.0), "19:42");
}

#[test]
fn sizes_use_the_largest_fitting_unit() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1023), "1023 B");
    assert_eq!(format_bytes(1024), "1.0 KB");
    assert_eq!(format_bytes(53_555), "52.3 KB");
    assert_eq!(format_bytes(9_437_184), "9.0 MB");
}

#[test]
fn variants_option_matches_the_pipeline() {
    use clap::CommandFactory;
    let command = Cli::command();
    let variants = command
        .get_arguments()
        .find(|arg| arg.get_id() == "variants")
        .expect("a variants option exists");
    assert_eq!(
        variants.get_default_values(),
        [std::ffi::OsStr::new(
            &pipeline::DEFAULT_VARIANTS.to_string()
        )]
    );

    // The highest allowed value is accepted, one above it is refused.
    let max = pipeline::VARIANTS.len();
    assert!(Cli::try_parse_from(["transcribe", "a.mp3", "--variants", &max.to_string()]).is_ok());
    assert!(
        Cli::try_parse_from(["transcribe", "a.mp3", "--variants", &(max + 1).to_string()]).is_err()
    );
    assert!(Cli::try_parse_from(["transcribe", "a.mp3", "--variants", "0"]).is_err());
}
