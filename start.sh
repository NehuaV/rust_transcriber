#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Check if the user specified a binary mode explicitly
if [ "$1" = "transcribe" ] || [ "$1" = "basic" ]; then
    TARGET="transcribe"
    shift
elif [ "$1" = "stranscribe" ] || [ "$1" = "streaming" ]; then
    TARGET="stranscribe"
    shift
else
    # Default engine is stranscribe
    TARGET="stranscribe"
fi

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
