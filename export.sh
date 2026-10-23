#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TARGET_DIR="${1:-$HOME/.local/bin}"
SHARE_DIR="$HOME/.local/share/transcriber"

# 1. Compile release binaries
echo "Building release binaries..."
(cd "$DIR" && cargo build --release)

# 2. Ensure target and share directories exist
mkdir -p "$TARGET_DIR"
mkdir -p "$SHARE_DIR"

# 3. Install the compiled binaries directly into TARGET_DIR
echo "Installing binaries to $TARGET_DIR..."
cp -f "$DIR/target/release/transcribe" "$TARGET_DIR/transcribe"
cp -f "$DIR/target/release/stranscribe" "$TARGET_DIR/stranscribe"
chmod +x "$TARGET_DIR/transcribe" "$TARGET_DIR/stranscribe"

# Sign binaries on macOS
if command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$TARGET_DIR/transcribe" >/dev/null 2>&1 || true
    codesign -s - -f "$TARGET_DIR/stranscribe" >/dev/null 2>&1 || true
fi

# 4. Link shared models to ~/.local/share/transcriber so binaries resolve them independently
ln -sfn "$DIR/models" "$SHARE_DIR/models"

# 5. Remove root copies from the repository so it remains clean
rm -f "$DIR/transcribe" "$DIR/stranscribe"

# 6. Detect user shell configuration file
DETECTED_SHELL="${SHELL##*/}"
case "$DETECTED_SHELL" in
    zsh)
        CONFIG_FILE="$HOME/.zshrc"
        ;;
    bash)
        if [ -f "$HOME/.bashrc" ]; then
            CONFIG_FILE="$HOME/.bashrc"
        else
            CONFIG_FILE="$HOME/.bash_profile"
        fi
        ;;
    *)
        CONFIG_FILE="$HOME/.profile"
        ;;
esac

echo ""
echo "Installation complete."
echo "Binaries installed in: $TARGET_DIR"
echo "  - $TARGET_DIR/transcribe"
echo "  - $TARGET_DIR/stranscribe"
echo ""

# Check if TARGET_DIR is in PATH
case ":$PATH:" in
    *":$TARGET_DIR:"*)
        ;;
    *)
        echo "Note: '$TARGET_DIR' was not found in your current PATH."
        echo "Add it by appending this line to $CONFIG_FILE:"
        echo "  export PATH=\"$TARGET_DIR:\$PATH\""
        echo ""
        ;;
esac

echo "Please reload your shell configuration by running:"
echo "  source $CONFIG_FILE"
