#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BIN="$DIR/transcriber"

if [ ! -f "$BIN" ]; then
    if [ -f "$DIR/target/release/transcriber" ]; then
        BIN="$DIR/target/release/transcriber"
    else
        echo "Transcriber binary not found. Building release binary..."
        (cd "$DIR" && cargo build --release)
        BIN="$DIR/target/release/transcriber"
    fi
fi

exec "$BIN" "$@"
