"""dekupe: video background removal with the audited RVM ONNX model (own inference code).

Frames come from ffmpeg as raw RGB, go through the recurrent ONNX model one at a time, and are
written to one or more ffmpeg encoders in a single pass.

Outputs (any combination, at least one):
  --alpha  OUT.mov   ProRes 4444 with alpha (Premiere / Resolve)
  --webm   OUT.webm  VP9 with alpha (Remotion <OffthreadVideo transparent>)
  --green  OUT.mp4   subject over flat green (--key-color)
  --bg SRC --out OUT.mp4   subject over an image, a video (looped) or a #hex color
  --matte  OUT.mp4   grey alpha matte (white = subject), same frame count as the input

Example:
  python -I matte.py in.mp4 --alpha cut.mov --matte alpha.mp4 --bg beach.jpg --out beach.mp4
"""
import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")
PINS = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
Q = ["-hide_banner", "-loglevel", "error"]


# ---------- probing ----------
def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation:stream_tags=rotate",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(s.get("tags", {}).get("rotate", 0) or 0)
    if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode
        w, h = h, w
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip())
    return w, h, s["r_frame_rate"], int(s.get("nb_frames") or 0), has_audio


# ---------- model ----------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wrap_model(m, fp16, use_src):
    """Add our own pre/post-processing to the audited graph (standard opset-12 ops only):
    img uint8 [H,W,3] -> src [1,3,H,W] in 0..1  ...  fgr,pha -> rgba uint8 [H,W,4].
    Keeps the heavy per-pixel work on the GPU so only uint8 frames cross the bus."""
    g = m.graph
    ft = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    for node in g.node:  # internal names
        node.input[:] = ["_src" if i == "src" else i for i in node.input]
        node.output[:] = [{"fgr": "_fgr", "pha": "_pha"}.get(o, o) for o in node.output]
    const = lambda name, v: helper.make_node("Constant", [], [name], value=helper.make_tensor(
        name + "_t", TensorProto.FLOAT, [], [v]))
    pre = [
        helper.make_node("Cast", ["img"], ["_img_f"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["_img_f"], ["_img_chw"], perm=[2, 0, 1]),
        helper.make_node("Unsqueeze", ["_img_chw"], ["_img_nchw"], axes=[0]),
        const("_inv255", 1.0 / 255.0),
        helper.make_node("Mul", ["_img_nchw", "_inv255"], ["_src_f"]),
        helper.make_node("Cast", ["_src_f"], ["_src"], to=ft),
    ]
    color = "_src" if use_src else "_fgr"
    post = [
        helper.make_node("Concat", [color, "_pha"], ["_rgba_h"], axis=1),
        helper.make_node("Cast", ["_rgba_h"], ["_rgba_f"], to=TensorProto.FLOAT),
        helper.make_node("Squeeze", ["_rgba_f"], ["_rgba_chw"], axes=[0]),
        helper.make_node("Transpose", ["_rgba_chw"], ["_rgba_hwc"], perm=[1, 2, 0]),
        const("_zero", 0.0), const("_one", 1.0), const("_k255", 255.0), const("_half", 0.5),
        helper.make_node("Clip", ["_rgba_hwc", "_zero", "_one"], ["_rgba_c"]),
        helper.make_node("Mul", ["_rgba_c", "_k255"], ["_rgba_s"]),
        helper.make_node("Add", ["_rgba_s", "_half"], ["_rgba_r"]),
        helper.make_node("Cast", ["_rgba_r"], ["rgba"], to=TensorProto.UINT8),
    ]
    nodes = pre + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    ins = [i for i in g.input if i.name != "src"]
    del g.input[:]
    g.input.extend([helper.make_tensor_value_info("img", TensorProto.UINT8, ["H", "W", 3])] + ins)
    outs = [o for o in g.output if o.name not in ("fgr", "pha")]
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("rgba", TensorProto.UINT8, ["H", "W", 4])] + outs)
    return m


class Matter:
    """Recurrent RVM runner. feed(uint8 HxWx3) -> uint8 HxWx4 (RGB = fgr or src, A = alpha)."""

    def __init__(self, variant, precision, device, use_src):
        if hasattr(ort, "preload_dlls"):  # CUDA/cuDNN from the nvidia-* pip wheels
            try:
                ort.preload_dlls()
            except Exception:
                pass
        avail = ort.get_available_providers()
        wanted = {"cuda": ["CUDAExecutionProvider"], "dml": ["DmlExecutionProvider"], "cpu": []}
        order = wanted[device] if device != "auto" else ["CUDAExecutionProvider", "DmlExecutionProvider"]
        providers = [p for p in order if p in avail] + ["CPUExecutionProvider"]
        if precision == "auto":
            precision = "fp32"
        if variant == "auto":  # same speed on GPU, better edges -> resnet50; light model on CPU
            variant = "mobilenetv3" if providers[0] == "CPUExecutionProvider" else "resnet50"
        self.name = f"rvm_{variant}_{precision}"
        path = os.path.join(MODELS_DIR, self.name + ".onnx")
        if not os.path.exists(path):
            sys.exit(f"model missing: {path}\nrun: python -I {os.path.join(HERE, 'fetch_model.py')} {self.name}")
        if sha256(path) != PINS[self.name]["sha256"]:
            sys.exit(f"REFUSING {path}: sha256 does not match the audited pin in models.json")
        m = wrap_model(onnx.load(path, load_external_data=False), precision == "fp16", use_src)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(m.SerializeToString(), sess_options=so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.dt = np.float16 if precision == "fp16" else np.float32
        self.dev = "cuda" if self.provider == "CUDAExecutionProvider" else None
        self.rec = None
        self.drt = None

    def reset(self, dr):
        z = np.zeros((1, 1, 1, 1), self.dt)
        d = np.array([dr], np.float32)
        if self.dev:
            self.rec = [ort.OrtValue.ortvalue_from_numpy(z, self.dev, 0) for _ in range(4)]
            self.drt = ort.OrtValue.ortvalue_from_numpy(d, self.dev, 0)
        else:
            self.rec, self.drt = [z] * 4, d

    def feed(self, frame):
        names = ["r1i", "r2i", "r3i", "r4i"]
        if not self.dev:
            rgba, *self.rec = self.sess.run(None, {"img": frame, **dict(zip(names, self.rec)),
                                                   "downsample_ratio": self.drt})
            return rgba
        io = self.sess.io_binding()
        io.bind_cpu_input("img", frame)
        for n, v in zip(names, self.rec):
            io.bind_ortvalue_input(n, v)
        io.bind_ortvalue_input("downsample_ratio", self.drt)
        io.bind_output("rgba", "cpu")
        for n in ["r1o", "r2o", "r3o", "r4o"]:
            io.bind_output(n, self.dev)
        self.sess.run_with_iobinding(io)
        outs = io.get_outputs()
        self.rec = outs[1:]
        return outs[0].numpy()


# ---------- io ----------
def reader(path, w, h, max_frames):
    cmd = ["ffmpeg", *Q, "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)


def read_frame(proc, n):
    buf = proc.stdout.read(n)
    return buf if len(buf) == n else None


_NVENC = None


def nvenc_ok():
    global _NVENC
    if _NVENC is None:
        _NVENC = subprocess.run(["ffmpeg", *Q, "-f", "lavfi", "-i", "color=s=256x256", "-frames:v", "1",
                                 "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True).returncode == 0
    return _NVENC


def h264(quality):
    if nvenc_ok():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(quality),
                "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", str(quality - 2), "-pix_fmt", "yuv420p"]


ENC = {
    "alpha": ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"],
    "webm": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "20", "-row-mt", "1",
             "-deadline", "good", "-cpu-used", "4", "-tile-columns", "2", "-threads", "16",
             "-auto-alt-ref", "0", "-metadata:s:v:0", "alpha_mode=1"],
    "matte": ["-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p"],
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def bg_input(spec, w, h, fps):
    """ffmpeg input args for a background: #hex color, still image, or video (looped)."""
    if spec.startswith("#"):
        hx = spec[1:]
        if len(hx) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hx):
            sys.exit(f"bad color {spec!r}, use #RRGGBB")
        return ["-f", "lavfi", "-i", f"color=c=0x{hx}:s={w}x{h}:r={fps}"]
    if not os.path.exists(spec):
        sys.exit(f"background not found: {spec}")
    if spec.lower().endswith(IMAGE_EXT):
        return ["-loop", "1", "-framerate", fps, "-i", spec]
    return ["-stream_loop", "-1", "-i", spec]


def writer(out, pix_fmt, w, h, fps, vcodec, src, audio, bg=None):
    """ffmpeg encoder reading raw frames on stdin. With bg, ffmpeg itself composites the RGBA
    frames over the background (overlay = straight-alpha blend)."""
    cmd = ["ffmpeg", *Q, "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", fps, "-i", "-"]
    vmap, nxt = "0:v", 1
    if bg:
        cmd += bg_input(bg, w, h, fps)
        cmd += ["-filter_complex",
                f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
                f"format=rgba[b];[b][0:v]overlay=shortest=1:format=auto,format=yuv420p[v]"]
        vmap, nxt = "[v]", 2
    if audio:  # all inputs first, then output options
        cmd += ["-i", src]
    cmd += ["-map", vmap]
    if audio:
        cmd += ["-map", f"{nxt}:a:0?", "-shortest"]
    cmd += vcodec
    if audio:
        cmd += ["-c:a", "libopus" if out.lower().endswith(".webm") else "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += [out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Output(threading.Thread):
    """One encoder fed from its own thread so encoders run in parallel."""

    def __init__(self, kind, proc):
        super().__init__(daemon=True)
        self.kind, self.proc = kind, proc
        self.q = queue.Queue(maxsize=6)
        self.error = None
        self.start()

    def run(self):
        try:
            while (rgba := self.q.get()) is not None:
                data = rgba if self.kind == "rgba" else np.ascontiguousarray(rgba[..., 3])
                self.proc.stdin.write(data.tobytes())
        except Exception as e:  # broken pipe = encoder died
            self.error = e
            while self.q.get() is not None:  # drain so the producer never blocks
                pass

    def finish(self):
        self.q.put(None)
        self.join()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        code = self.proc.wait()
        return code if not self.error else (code or 1)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--alpha", help="ProRes 4444 .mov with alpha")
    ap.add_argument("--webm", help="VP9 .webm with alpha")
    ap.add_argument("--green", help="MP4 over flat key color")
    ap.add_argument("--key-color", default="#00B140")
    ap.add_argument("--bg", help="background for --out: image, video or #hex")
    ap.add_argument("--out", help="MP4 composited over --bg")
    ap.add_argument("--matte", help="grey alpha matte MP4")
    ap.add_argument("--model", choices=["auto", "mobilenetv3", "resnet50"], default="auto",
                    help="auto = resnet50 on GPU, mobilenetv3 on CPU")
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto = fp32 (measured faster than fp16 on CUDA / RTX 5070)")
    ap.add_argument("--device", choices=["auto", "cuda", "dml", "cpu"], default="auto")
    ap.add_argument("--dr", type=float, help="downsample_ratio (default: auto from resolution)")
    ap.add_argument("--full-body", action="store_true", help="auto dr for full-body shots (higher)")
    ap.add_argument("--use-src", action="store_true", help="use original pixels instead of RVM fgr for color")
    ap.add_argument("--max-frames", type=int, default=0, help="only the first N frames (previews)")
    a = ap.parse_args()

    if bool(a.bg) != bool(a.out):
        ap.error("--bg and --out go together")
    if not any([a.alpha, a.webm, a.green, a.out, a.matte]):
        ap.error("give at least one output: --alpha/--webm/--green/--bg+--out/--matte")
    if not os.path.exists(a.input):
        ap.error(f"input not found: {a.input}")

    w, h, fps, nb, has_audio = probe(a.input)
    audio = has_audio and not a.max_frames
    dr = a.dr or min(1.0, (768.0 if a.full_body else 480.0) / max(w, h))
    mt = Matter(a.model, a.precision, a.device, a.use_src)
    mt.reset(dr)
    print(f"[dekupe] {a.input}: {w}x{h} @ {fps}  frames~{nb or '?'}  model={mt.name}  "
          f"provider={mt.provider}  dr={dr:.3f}", flush=True)

    outs = []
    if a.alpha:
        outs.append(Output("rgba", writer(a.alpha, "rgba", w, h, fps, ENC["alpha"], a.input, audio)))
    if a.webm:
        outs.append(Output("rgba", writer(a.webm, "rgba", w, h, fps, ENC["webm"], a.input, audio)))
    if a.green:
        outs.append(Output("rgba", writer(a.green, "rgba", w, h, fps, h264(18), a.input, audio, a.key_color)))
    if a.out:
        outs.append(Output("rgba", writer(a.out, "rgba", w, h, fps, h264(18), a.input, audio, a.bg)))
    if a.matte:
        outs.append(Output("matte", writer(a.matte, "gray", w, h, fps, ENC["matte"], a.input, False)))

    # decode in a thread too, so ffmpeg decode, GPU inference and encoders overlap
    rd = reader(a.input, w, h, a.max_frames)
    frames = queue.Queue(maxsize=6)

    def decode():
        nbytes = w * h * 3
        while (buf := read_frame(rd, nbytes)) is not None:
            frames.put(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        frames.put(None)

    threading.Thread(target=decode, daemon=True).start()
    n, t0 = 0, time.time()
    try:
        while (frame := frames.get()) is not None:
            rgba = mt.feed(frame)
            for o in outs:
                o.q.put(rgba)
            n += 1
            if n % 100 == 0:
                print(f"[dekupe] {n}/{nb or '?'} frames  {n / (time.time() - t0):.1f} fps", flush=True)
    finally:
        rd.stdout.close()
        rd.wait()
        codes = [o.finish() for o in outs]
    secs = time.time() - t0
    print(f"[dekupe] done: {n} frames in {secs:.1f}s ({n / max(secs, 1e-6):.1f} fps)", flush=True)
    if any(codes):
        sys.exit(f"an encoder failed (exit codes {codes})")
    expected = a.max_frames or nb
    if expected and n != expected:
        print(f"[dekupe] WARNING: read {n} frames but the input reports {expected}", flush=True)


if __name__ == "__main__":
    main()
