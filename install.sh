#!/usr/bin/env sh
set -eu

REPO="YOUR_USERNAME/djeli"
BASE="https://github.com/${REPO}/releases/latest/download"
BIN="djeli"
INSTALL_DIR="${DJELI_INSTALL_DIR:-$HOME/.local/bin}"

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

echo "Installed ${BIN} to ${INSTALL_DIR}/${BIN}"
echo "Next: mkdir -p ~/.djeli/models, then download a .gguf file, then run: djeli serve"