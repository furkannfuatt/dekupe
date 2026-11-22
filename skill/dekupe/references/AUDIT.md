# Quarantine audit — RobustVideoMatting (2026-11-22)

Tooling: REA `rea-agents` 6.5.0 CLI (static, no target execution), `grep`, our `scripts/audit_onnx.py`.
Quarantine: a separate empty folder per download (raw REA JSON output kept locally, not committed). No file from the repo was executed at any point.

## 1. Source archive
- `master.zip` from github.com/PeterL1n/RobustVideoMatting, SHA-256
  `c443968f4dab2d17612734a3822fc9f1d39d450abcb07e63f38548790576bc16`.
- REA `inspect-artifact` — evidence `ev_969569f0a433b965…`, inventory `ev_8ba762d5f490c604…`:
  52 central-directory entries, **0 executable, 0 encrypted**, all hashes verified, no integrity
  contradictions, no hypotheses/next probes. Contents: 28 `.py`, docs/markdown, 2 GIFs, txt lists.
  **No model weights or binaries inside the repo.**
- REA `extract-artifact` — evidence `ev_19cd1ce0aaae488b…`, `containment_verified: true` (no path escape).

## 2. Source review (read as text only)
grep `subprocess|os.system|eval(|exec(|urllib|requests|socket|pickle|torch.load|__reduce__|base64|download|http`:
- **Observed, no hits** for subprocess, os.system, exec, socket, requests, urllib, base64, `__reduce__`.
- **Observed risk (pickle loading):**
  - `hubconf.py:20,28` — `torch.hub.load_state_dict_from_url` of `rvm_mobilenetv3.pth` / `rvm_resnet50.pth`.
  - `model/mobilenetv3.py:30`, `model/resnet.py:14` — download ImageNet `.pth` from download.pytorch.org.
  - `inference.py:163`, `train.py:324` — `torch.load(checkpoint)` without `weights_only=True`.
  `.pth` = pickle; loading one runs whatever the file says. `torch.hub.load(...)` additionally imports and
  runs `hubconf.py` from GitHub at call time (the code could change after this audit).
- `eval` hits are `model.eval()` (PyTorch mode switch), not Python `eval`.
- **Inference (not observed):** nothing in the source looks malicious; the danger is the execution model
  (remote code + pickle), not specific lines. We avoid both entirely.

## 3. ONNX models (GitHub release v1.0.0)
REA 6.5.0 does **not** support ONNX: `inspect-artifact` → `unsupported_target / unsupported binary format`
(recorded as a tool limitation, not a finding). Checked instead with `audit_onnx.py` (protobuf parse only,
`load_external_data=False`, `onnx.checker`, no session):

| model | SHA-256 | size | nodes | result |
|---|---|---|---|---|
| rvm_mobilenetv3_fp32 | 88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828 | 14975696 | 353 | PASS |
| rvm_mobilenetv3_fp16 | 6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64 | 7503483 | 353 | PASS |
| rvm_resnet50_fp32 | 25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c | 107479165 | 313 | PASS |
| rvm_resnet50_fp16 | a9266f5046411d604bbff38e18c54bf8c70d85d93bbc400564697590f5712738 | 53752431 | 313 | PASS |

All: producer pytorch 1.9, opset `ai.onnx` 12 only, no custom domains, no external data, no embedded
functions/training info; I/O exactly `src,r1i..r4i,downsample_ratio → fgr,pha,r1o..r4o`. Only math/tensor
ops (Conv, Resize, Sigmoid, Tanh, …). An ONNX graph of standard ops cannot run arbitrary code; the residual
trust is in onnxruntime itself.

## 4. Decision
Use only the four pinned ONNX files with our own `matte.py`. Never torch.hub / `.pth` / repo `.py`.
Tamper test: flipping one bit makes `fetch_model.py --verify` and `matte.py` refuse the file (verified).

## Unknowns
- Whether the ONNX export matches the published `.pth` exactly (not needed for our use; quality checked
  visually on the user's footage).
- The repo is GPL-3.0; weights inherit that license.
