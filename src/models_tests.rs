use super::*;

/// A fresh empty folder under the system temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = env::temp_dir().join(format!("transcriber-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_first_complete_folder_wins() {
    let incomplete = TempDir::new("incomplete");
    let first = TempDir::new("first");
    let second = TempDir::new("second");
    fs::write(first.0.join(MARKER), "{}").unwrap();
    fs::write(second.0.join(MARKER), "{}").unwrap();

    let found = find(vec![
        incomplete.0.clone(),
        first.0.clone(),
        second.0.clone(),
    ]);
    assert_eq!(found, Some(first.0.clone()));
}

#[test]
fn no_complete_folder_finds_nothing() {
    let empty = TempDir::new("empty");
    assert_eq!(
        find(vec![empty.0.clone(), PathBuf::from("/does/not/exist")]),
        None
    );
}

#[test]
fn candidates_end_with_the_project_models_folder() {
    let places = candidates();
    assert_eq!(
        places.last().unwrap(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("models/coreml")
    );
    assert!(places.contains(&PathBuf::from("models/coreml")));
}
