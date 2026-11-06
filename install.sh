#!/usr/bin/env sh
set -eu

REPO="bigdahaut/djeli"
BASE="https://github.com/${REPO}/releases/latest/download"
BIN="djeli"
INSTALL_DIR="${DJELI_INSTALL_DIR:-$HOME/.local/bin}"
MODELS_DIR="$HOME/.djeli/models"

OS="$(uname -s)"
case "$OS" in
  Linux)  os="linux"  ;;
  Darwin) os="macos"  ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="aarch64" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

URL="${BASE}/djeli-${os}-${arch}"
echo "Downloading ${URL}…"
mkdir -p "$INSTALL_DIR"
curl -fsSL "$URL" -o "${INSTALL_DIR}/${BIN}"
chmod +x "${INSTALL_DIR}/${BIN}"

# Automatically create the models directory
mkdir -p "$MODELS_DIR"

echo "Installed ${BIN} to ${INSTALL_DIR}/${BIN}"
echo "Created models directory at ${MODELS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Download a .gguf model and place it in: ${MODELS_DIR}"
echo "  2. Start the engine: djeli serve"
echo "  3. Run the model: djeli run <model-name>"
