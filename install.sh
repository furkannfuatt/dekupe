#!/usr/bin/env bash
# dekupe installer (Linux / macOS). Copies the skill to ~/.claude/skills/dekupe, creates its venv,
# downloads the pinned ONNX models and runs the static audit.
#   ./install.sh          # auto: CUDA build on Linux with nvidia-smi, CPU build otherwise
#   ./install.sh cpu
set -euo pipefail

for tool in python3 ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "$tool not found on PATH" >&2; exit 1; }
done

runtime="${1:-auto}"
if [ "$runtime" = auto ]; then
  if [ "$(uname)" = Linux ] && command -v nvidia-smi >/dev/null; then runtime=cuda; else runtime=cpu; fi
fi
case "$runtime" in
  cuda) ort="onnxruntime-gpu[cuda,cudnn]" ;;
  cpu)  ort="onnxruntime" ;;
  *) echo "usage: $0 [auto|cuda|cpu]" >&2; exit 1 ;;
esac

dest="$HOME/.claude/skills/dekupe"
src="$(cd "$(dirname "$0")" && pwd)/skill/dekupe"
mkdir -p "$dest"
cp -R "$src/." "$dest/"

py="$dest/.venv/bin/python"
[ -x "$py" ] || python3 -m venv "$dest/.venv"
"$py" -m pip install --quiet --upgrade pip
"$py" -m pip install --quiet numpy onnx "$ort"

"$py" -I "$dest/scripts/fetch_model.py" --all
"$py" -I "$dest/scripts/audit_onnx.py" "$dest"/models/*.onnx

echo
echo "dekupe installed to $dest"
echo "Try: $py -I $dest/scripts/matte.py input.mp4 --green green.mp4"
