"""Static safety check for an RVM ONNX model. Never runs the model.

Parses the protobuf only (no external data, no session) and fails unless:
  - every node uses a standard domain ("" / ai.onnx) and no custom opset is imported
  - no tensor references external data, no functions or training info are embedded
  - inputs/outputs match the RVM contract (src, r1i..r4i, downsample_ratio -> fgr, pha, r1o..r4o)

Usage: python -I audit_onnx.py MODEL.onnx [MODEL2.onnx ...]   (exit 0 = all pass)
"""
import collections
import hashlib
import sys

import onnx

STD_DOMAINS = {"", "ai.onnx"}
EXPECTED_IN = ["src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"]
EXPECTED_OUT = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def all_graphs(g):
    yield g
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from all_graphs(a.g)
            for sg in a.graphs:
                yield from all_graphs(sg)


def audit(path):
    problems = []
    m = onnx.load(path, load_external_data=False)
    opsets = {o.domain or "": o.version for o in m.opset_import}
    for d in opsets:
        if d not in STD_DOMAINS:
            problems.append(f"non-standard opset domain imported: {d!r}")
    if len(m.functions):
        problems.append(f"model embeds {len(m.functions)} local functions")
    if len(m.training_info):
        problems.append("model embeds training_info")

    ops = collections.Counter()
    n_init = 0
    for g in all_graphs(m.graph):
        for n in g.node:
            if (n.domain or "") not in STD_DOMAINS:
                problems.append(f"node {n.name!r} uses custom domain {n.domain!r} ({n.op_type})")
            ops[n.op_type] += 1
        for t in g.initializer:
            n_init += 1
            if t.data_location == onnx.TensorProto.EXTERNAL or len(t.external_data):
                problems.append(f"initializer {t.name!r} references external data")

    ins = [i.name for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    outs = [o.name for o in m.graph.output]
    if ins != EXPECTED_IN:
        problems.append(f"inputs {ins} != {EXPECTED_IN}")
    if outs != EXPECTED_OUT:
        problems.append(f"outputs {outs} != {EXPECTED_OUT}")

    try:
        onnx.checker.check_model(m)
    except Exception as e:  # structural problem
        problems.append(f"onnx.checker: {e}")

    def dtype(v):
        return onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)

    print(f"== {path}")
    print(f"   sha256   {sha256(path)}")
    print(f"   producer {m.producer_name} {m.producer_version}  opsets {opsets}  ir {m.ir_version}")
    print(f"   inputs   {[(i.name, dtype(i)) for i in m.graph.input]}")
    print(f"   outputs  {[(o.name, dtype(o)) for o in m.graph.output]}")
    print(f"   nodes {sum(ops.values())}  initializers {n_init}  op types {len(ops)}: {', '.join(sorted(ops))}")
    for p in problems:
        print(f"   FAIL {p}")
    print("   PASS" if not problems else "   REJECTED")
    return not problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok = all([audit(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
