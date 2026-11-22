"""Download pinned RVM ONNX models into ../models and verify SHA-256 + size.

A file whose hash does not match models.json is deleted and the script fails.
Usage: python -I fetch_model.py [name ...]      (default: rvm_mobilenetv3_fp16 rvm_mobilenetv3_fp32)
       python -I fetch_model.py --all
       python -I fetch_model.py --verify        (only check files already in ../models)
"""
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")
PINS = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
PINS.pop("_note", None)
DEFAULT = ["rvm_mobilenetv3_fp16", "rvm_mobilenetv3_fp32"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(name):
    path = os.path.join(MODELS_DIR, name + ".onnx")
    if not os.path.exists(path):
        return False, "missing"
    pin = PINS[name]
    if os.path.getsize(path) != pin["size"]:
        return False, f"size {os.path.getsize(path)} != {pin['size']}"
    digest = sha256(path)
    if digest != pin["sha256"]:
        return False, f"sha256 {digest} != pinned {pin['sha256']}"
    return True, "ok"


def fetch(name):
    ok, why = verify(name)
    if ok:
        print(f"{name}: already present, hash ok")
        return True
    path = os.path.join(MODELS_DIR, name + ".onnx")
    tmp = path + ".part"
    print(f"{name}: downloading ({why})")
    req = urllib.request.Request(PINS[name]["url"], headers={"User-Agent": "dekupe-fetch"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, path)
    ok, why = verify(name)
    if not ok:
        os.remove(path)
        print(f"{name}: REJECTED and deleted: {why}")
        return False
    print(f"{name}: ok")
    return True


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    args = sys.argv[1:]
    if args == ["--verify"]:
        results = []
        for n in PINS:
            ok, why = verify(n)
            if why != "missing":
                print(f"{n}: {why}")
                results.append(ok)
        sys.exit(0 if results and all(results) else 1)
    names = list(PINS) if args == ["--all"] else (args or DEFAULT)
    unknown = [n for n in names if n not in PINS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; known: {list(PINS)}")
    sys.exit(0 if all([fetch(n) for n in names]) else 1)
