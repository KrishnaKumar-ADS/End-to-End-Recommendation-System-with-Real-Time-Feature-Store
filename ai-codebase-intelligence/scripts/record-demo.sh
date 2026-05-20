#!/usr/bin/env bash

set -euo pipefail

OUTPUT_FILE="${1:-demo-raw.mp4}"
FPS="${FPS:-30}"
RESOLUTION="${RESOLUTION:-1920x1080}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed."
  exit 1
fi

OS_NAME="$(uname -s)"

echo "Recording demo to ${OUTPUT_FILE}"
echo "Press Ctrl+C to stop recording."

case "${OS_NAME}" in
  Darwin)
    ffmpeg \
      -f avfoundation \
      -framerate "${FPS}" \
      -video_size "${RESOLUTION}" \
      -i "1:none" \
      -c:v libx264 \
      -preset veryfast \
      -pix_fmt yuv420p \
      "${OUTPUT_FILE}"
    ;;
  Linux)
    DISPLAY_NAME="${DISPLAY:-:0.0}"
    ffmpeg \
      -f x11grab \
      -framerate "${FPS}" \
      -video_size "${RESOLUTION}" \
      -i "${DISPLAY_NAME}" \
      -c:v libx264 \
      -preset veryfast \
      -pix_fmt yuv420p \
      "${OUTPUT_FILE}"
    ;;
  *)
    echo "Unsupported OS for this helper: ${OS_NAME}"
    echo "Use your platform recording tool and export to MP4."
    exit 1
    ;;
esac
