"""逐层比对 numpy 参考实现与 onnxruntime，定位第一处语义偏差。"""
import sys
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, r"d:\MusicRecorder\.onnx_export")
from ref_eval import RefEvaluator, MODEL

TOL = 1e-3

m = onnx.load(MODEL)
g = m.graph
init_names = {i.name for i in g.initializer}
existing_out = {o.name for o in g.output}

# 追加所有中间张量为图输出
added = 0
seen = set()
for n in g.node:
    for on in n.output:
        if on == "" or on in init_names or on in existing_out or on in seen:
            continue
        seen.add(on)
        vi = onnx.helper.ValueInfoProto()
        vi.name = on
        g.output.append(vi)
        added += 1
print("追加中间输出:", added)

sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3
sess = ort.InferenceSession(m.SerializeToString(), sess_opts, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

rng = np.random.default_rng(1234)
audio = (rng.standard_normal((1, 43844, 1)).astype(np.float32) * 0.3)

# 我的实现：记录全部 env
ev = RefEvaluator(MODEL)
m2 = onnx.load(MODEL)
env = dict(ev.inits)
env[iname] = audio.astype(np.float32)

ort_names = [o.name for o in sess.get_outputs()]
ort_vals = sess.run(ort_names, {iname: audio})
ort_map = dict(zip(ort_names, ort_vals))

first_bad = None
for idx, n in enumerate(ev.nodes):
    ins = [env[i] if i != "" else None for i in n.input]
    attrs = {}
    for a in n.attribute:
        if a.type == onnx.AttributeProto.INT:
            attrs[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            attrs[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            attrs[a.name] = a.s
        elif a.type == onnx.AttributeProto.INTS:
            attrs[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            attrs[a.name] = list(a.floats)
    try:
        out = ev._eval(n.op_type, ins, attrs)
    except Exception as e:
        print("节点 %d %s 求值异常: %s" % (idx, n.op_type, e))
        break
    for on in n.output:
        if on == "":
            continue
        env[on] = out
    # 比对主输出（n.output[0]）
    on0 = n.output[0]
    if on0 in ort_map:
        a = ort_map[on0]
        b = np.asarray(out)
        if a.shape == b.shape:
            a64 = a.astype(np.float64)
            b64 = b.astype(np.float64)
            d = float(np.max(np.abs(a64 - b64)))
            scale = max(1.0, float(np.max(np.abs(a64))))
            rel = d / scale
            if rel > 0.01:
                print("偏差 @ 节点 %d %-14s name=%s shape=%s maxdiff=%.4e rel=%.3e" %
                      (idx, n.op_type, on0, a.shape, d, rel))
                print("   ORT: ", a.flatten()[:5])
                print("   ref: ", b.flatten()[:5])
        else:
            print("形状不一致 @ 节点 %d %s: ORT %s vs ref %s" % (idx, n.op_type, a.shape, b.shape))

print("=== 最终输出比对 ===")
for o in sess.get_outputs():
    if o.name in env and o.name in ort_map:
        a = ort_map[o.name].astype(np.float64)
        b = np.asarray(env[o.name]).astype(np.float64)
        print("  %-28s maxdiff=%.4e  max|ort|=%.5f" % (o.name, np.max(np.abs(a - b)), np.max(np.abs(a))))