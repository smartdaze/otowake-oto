#!/bin/bash
# ONNXモデルをGitHub Releasesからダウンロードするセットアップスクリプト
#
# 使い方: npm run setup:models
#         または bash scripts/setup-models.sh

set -e

REPO="smartdaze/otowake-oto"
TAG="models-v1"
MODEL_DIR="public/models"

MODELS=(
  "htdemucs_6s.onnx"
  "htdemucs_6s.json"
)

mkdir -p "$MODEL_DIR"

for file in "${MODELS[@]}"; do
  dest="$MODEL_DIR/$file"
  if [ -f "$dest" ]; then
    echo "[skip] $file already exists"
    continue
  fi

  url="https://github.com/$REPO/releases/download/$TAG/$file"
  echo "[download] $file from $url"

  if command -v curl &>/dev/null; then
    curl -L --fail --progress-bar -o "$dest" "$url"
  elif command -v wget &>/dev/null; then
    wget -q --show-progress -O "$dest" "$url"
  else
    echo "ERROR: curl or wget required" >&2
    exit 1
  fi

  echo "[done] $file ($(du -h "$dest" | cut -f1))"
done

echo ""
echo "All models ready in $MODEL_DIR/"
