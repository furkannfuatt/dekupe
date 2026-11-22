# RVM ONNX I/O contract (our notes, from documentation/inference.md + audit_onnx.py output)

Inputs, in order:
- `src` float [B,3,H,W], RGB 0..1 (dtype = model precision)
- `r1i r2i r3i r4i` recurrent states (4 ConvGRU layers), dtype = model precision. First frame: zeros of
  shape [1,1,1,1]; afterwards feed back the previous frame's `r1o..r4o`. Frames must go in order.
- `downsample_ratio` float32 [1], always FP32 even in the fp16 model.

Outputs: `fgr` [B,3,H,W] foreground color 0..1 (edge spill removed), `pha` [B,1,H,W] alpha 0..1,
`r1o..r4o` next states.

Downsample ratio (doc table, portrait / full body):

| resolution | portrait | full body |
|---|---|---|
| <= 512x512 | 1 | 1 |
| 1280x720 | 0.375 | 0.6 |
| 1920x1080 | 0.25 | 0.4 |
| 3840x2160 | 0.125 | 0.2 |

Rule of thumb: downsampled image between 256 and 512 px. matte.py auto: dr = 480 / long side
(portrait column exactly), `--full-body`: 768 / long side (full-body column exactly).

Opset 12, exported by PyTorch 1.9. Op types: Add, AveragePool, Clip, Concat, Constant, Conv, Div, Expand,
GlobalAveragePool, HardSigmoid, MaxPool, Mul, ReduceMean, Relu, Resize, Shape, Sigmoid, Slice, Split,
Sub, Tanh — all standard `ai.onnx`.

Our in-memory wrapper (matte.py `wrap_model`) adds: `img` uint8 [H,W,3] → Cast/Transpose/Unsqueeze/Mul
→ `src`, and `fgr|src`+`pha` → Concat/Squeeze/Transpose/Clip/Mul/Add/Cast → `rgba` uint8 [H,W,4].
Recurrent states stay on the GPU via IOBinding.
