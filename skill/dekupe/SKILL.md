---
name: dekupe
description: Local, audited video background removal (dekupe / arka plan kesme / cut-out / green screen / alpha matte) with the Robust Video Matting ONNX model and our own inference code — no torch.hub, no repo code, no cloud. Use when the user wants a person cut out of a video, a transparent video (ProRes 4444 .mov / VP9 alpha .webm), a green-screen MP4, the background replaced with an image/video/color, or a grey alpha matte (e.g. for depth/cut-out layers in a reels edit). Triggers: "arka planı kes", "arkaplanı sil", "dekupe et", "yeşil ekran yap", "şeffaf video", "maske çıkar", "cut-out", "remove background from video", "green screen this".
---

# dekupe — safe local video matting

Everything runs on the user's machine (NVIDIA CUDA if available, else DirectML / CPU). The model is the
official RVM ONNX export, statically audited and pinned by SHA-256 (see `references/AUDIT.md`).
Inference is our own code in `scripts/matte.py`.

## Safety rules (non-negotiable)
- Never use `torch.hub`, never run any `.py` from the RobustVideoMatting repo, never `torch.load` a `.pth`
  (pickle = arbitrary code). Only the hash-pinned `.onnx` files in `models/` are allowed.
- `matte.py` re-checks the SHA-256 on every run and refuses a mismatching file. Never edit a hash in
  `scripts/models.json` without redoing the audit (`references/AUDIT.md`).
- No cloud matting services for the user's footage.

## Paths
```
SK=~/.claude/skills/dekupe
PY=$SK/.venv/Scripts/python      # Windows   (macOS/Linux: $SK/.venv/bin/python) — always run with -I
```

## First-time setup (only if `.venv` or `models/` is missing)
Run the repo's `install.ps1` (Windows) or `install.sh` (Linux/macOS), or manually:
```bash
python -m venv $SK/.venv
$PY -m pip install numpy onnx "onnxruntime-gpu[cuda,cudnn]"   # NVIDIA; otherwise: onnxruntime (CPU) or onnxruntime-directml
$PY -I $SK/scripts/fetch_model.py --all            # downloads + verifies pinned hashes, deletes mismatches
$PY -I $SK/scripts/audit_onnx.py $SK/models/*.onnx  # static check, must print PASS for each
```
Requires `ffmpeg`/`ffprobe` on PATH. Check the log line `provider=…`: `CUDAExecutionProvider` is the fast
path; `CPUExecutionProvider` works (~12 fps at 1080p with mobilenetv3).

## Usage
```bash
$PY -I $SK/scripts/matte.py INPUT.mp4 [outputs...] [options]
```
Outputs (combine freely, all produced in one pass, same frame count as the input, audio copied except matte):

| flag | file | use |
|---|---|---|
| `--alpha X.mov` | ProRes 4444 + alpha | Premiere / Resolve / Final Cut overlay track |
| `--webm X.webm` | VP9 + alpha | Remotion `<OffthreadVideo transparent>`, web |
| `--green X.mp4` | subject on #00B140 (`--key-color #RRGGBB`) | any editor, key later |
| `--bg IMG\|VID\|#hex --out X.mp4` | new background (cover-scaled; videos loop) | finished clip |
| `--matte X.mp4` | grey matte, white = subject | Track Matte / alphamerge |

Options:
- `--model auto|resnet50|mobilenetv3` — auto = resnet50 on GPU (same speed, better hair/edges), mobilenetv3 on CPU.
- `--dr R` downsample ratio. Auto = long side scaled to 480 px (1080x1920 → 0.25). `--full-body` uses
  768 px (→ 0.4) for full-body/wide shots. For close-ups with fine/backlit hair try `--dr 0.35`.
  Too low = mushy edges; higher is not always better.
- `--use-src` keep the original pixels for color instead of RVM's decontaminated foreground (`fgr`).
  Default `fgr` removes background color spill at the edges; use `--use-src` if the cut-out must match
  another layer of the same footage pixel-for-pixel, or if colors look off.
- `--max-frames N` quick preview of the first N frames (no audio).
- `--precision`, `--device` rarely needed (fp32 measured faster than fp16 on CUDA).

Examples:
```bash
$PY -I $SK/scripts/matte.py in.mp4 --alpha cut.mov --matte alpha.mp4      # editor: transparent ProRes + matte
$PY -I $SK/scripts/matte.py in.mp4 --bg beach.jpg --out beach.mp4         # new background
$PY -I $SK/scripts/matte.py in.mp4 --max-frames 50 --green preview.mp4    # 2-second preview
```

## Workflow
1. Never write outputs next to the user's originals unless asked; use a work folder.
2. Look at one frame of the source first. **Sideways footage** (landscape file with the person lying on
   their side and no rotation tag): rotate first — `ffmpeg -i in.mp4 -vf transpose=1 -c:v libx264 -crf 14 -c:a copy up.mp4`
   (`transpose=1` when the head is on the left, `transpose=2` when on the right). RVM expects upright people.
3. For long videos, run a `--max-frames 50 --green` preview, extract a frame
   (`ffmpeg -i preview.mp4 -vf "select=eq(n\,25)" -frames:v 1 -update 1 f.jpg`) and look at hair, hands
   and held objects; adjust `--dr` / `--full-body` / `--use-src`, then do the full run.
4. After the run: the log must say `done: N frames` equal to the input. Check alpha with `ffprobe`
   (`.mov` → `yuva444p12le`; `.webm` → tag `alpha_mode=1`).

Speed (RTX 5070, 1080x1920): inference 50–55 fps; matte/green/ProRes keep up; VP9 alpha is the slowest
encoder (~19 fps; ~13 fps when all five outputs are requested together).

## Known limits
- RVM is trained for people. Objects the person holds are usually kept; objects lying around are not.
- The first ~10 frames can be softer while the recurrent state warms up; start a second earlier and trim.
- Hard cuts inside one file: the recurrent memory carries over; matte each shot separately if a cut
  leaves ghosting.

## License
RVM weights are GPL-3.0 (PeterL1n/RobustVideoMatting); this skill is GPL-3.0 as well.

## Files
- `scripts/matte.py` — inference + encoders. Wraps the audited graph in memory with uint8 pre/post
  processing nodes (standard ops only) so frames stay on the GPU; recurrent state via IOBinding.
- `scripts/fetch_model.py` — pinned download / `--verify`.
- `scripts/audit_onnx.py` — static ONNX check (standard domains only, no external data, RVM I/O).
- `scripts/models.json` — URLs + SHA-256 + sizes.
- `references/AUDIT.md` — quarantine audit report. `references/rvm-io.md` — model I/O contract.
