#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TARGET="transcribe"
BIN="$DIR/$TARGET"

if [ -f "$BIN" ]; then
    EXEC_BIN="$BIN"
elif [ -f "$DIR/target/release/$TARGET" ]; then
    EXEC_BIN="$DIR/target/release/$TARGET"
else
    echo "Binary '$TARGET' not found. Building release binary..."
    (cd "$DIR" && cargo build --release --bin "$TARGET")
    EXEC_BIN="$DIR/target/release/$TARGET"
fi

exec "$EXEC_BIN" "$@"
