"""将 nmp.onnx 导出为「常量折叠后的 IR + 权重 blob」。

由于推理时输入形状固定为 [1, 43844, 1]，图中所有 Shape/Slice/Concat/Cast/Where
构成的动态形状计算都可以在导出阶段常量折叠掉。导出的 IR 中每个节点都带具体形状，
C# 引擎无需任何动态形状逻辑。

产物（写到 .onnx_export/out/）：
  ir.json          拓扑 + 常量属性 + 形状 + 缓冲区分配
  weights.bin      所有常量张量的 float32 数据（按 offset 索引）
  test_input.bin   验证用输入音频（随机，固定种子）
  test_out_*.bin   验证用标准答案（onnxruntime 输出）
"""
import json
import os
import numpy as np
import onnx
import onnxruntime as ort

import sys
sys.path.insert(0, r"d:\MusicRecorder\.onnx_export")
from ref_eval import RefEvaluator, MODEL, conv_nd, onnx_pad

OUT_DIR = r"d:\MusicRecorder\.onnx_export\out"
os.makedirs(OUT_DIR, exist_ok=True)

IN_SHAPE = [1, 43844, 1]


def get_attrs(node):
    attrs = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            attrs[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            attrs[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            attrs[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.INTS:
            attrs[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            attrs[a.name] = list(a.floats)
    return attrs


def main():
    m = onnx.load(MODEL)
    g = m.graph
    ev = RefEvaluator(MODEL)

    iname = ev.input_name
    rng = np.random.default_rng(1234)
    audio = (rng.standard_normal(IN_SHAPE).astype(np.float32) * 0.3)

    # ---- 1) 完整跑一遍，记录所有张量形状 ----
    full = ev.run({iname: audio})
    shapes = ev.shapes
    print("张量形状记录数:", len(shapes))

    # ---- 2) 常量折叠（含静态形状折叠） ----
    const = dict(ev.inits)
    kept = []
    folded_count = 0
    for n in ev.nodes:
        # Shape 的输入依赖音频，但形状是静态的 -> 直接折叠为常量
        if n.op_type == "Shape" and n.input[0] and n.input[0] not in const:
            val = np.array(shapes[n.input[0]], dtype=np.int64)
            for on in n.output:
                if on:
                    const[on] = val
            folded_count += 1
            continue
        all_const = all((i in const) for i in n.input if i != "")
        if all_const:
            ins = [const[i] if i != "" else None for i in n.input]
            out = ev._eval(n.op_type, ins, get_attrs(n))
            for on in n.output:
                if on:
                    const[on] = out
            folded_count += 1
        else:
            kept.append(n)
    print("折叠节点数: %d, 保留节点数: %d" % (folded_count, len(kept)))

    # ---- 3) 为保留节点解析常量输入 -> 内联属性 ----
    ir_nodes = []
    # 常量张量（作为保留节点输入使用的）-> 权重 blob
    wblob = bytearray()
    wtensors = {}          # name -> {offset, count, shape}
    buffers = {}           # name -> buffer index
    buf_shapes = {}        # buffer index -> shape
    buf_used = {}          # buffer index -> 是否会被后续使用
    next_buf = 0

    # 输入张量占 buffer 0
    buffers[iname] = next_buf
    buf_shapes[next_buf] = list(shapes[iname])
    next_buf += 1

    def ensure_weight(name):
        if name in wtensors:
            return
        arr = np.ascontiguousarray(const[name], dtype=np.float32)
        off = len(wblob)
        wblob.extend(arr.tobytes())
        wtensors[name] = {"offset": off, "count": int(arr.size), "shape": list(arr.shape),
                          "name": name}

    def alloc(name):
        nonlocal next_buf
        if name in buffers:
            return buffers[name]
        buffers[name] = next_buf
        buf_shapes[next_buf] = [int(d) for d in shapes[name]]
        next_buf += 1
        return next_buf - 1

    for n in kept:
        op = n.op_type
        attrs = get_attrs(n)
        ins, outs, extra = [], [alloc(o) for o in n.output if o], {}

        if op == "Reshape":
            extra["shape"] = [int(v) for v in const[n.input[1]]]
            ins = [buffers[n.input[0]]]
        elif op == "Unsqueeze":
            axes = const[n.input[1]] if len(n.input) > 1 else np.array(attrs["axes"])
            extra["axes"] = [int(a) for a in axes]
            ins = [buffers[n.input[0]]]
        elif op == "Slice":
            extra["starts"] = [int(v) for v in const[n.input[1]]]
            extra["ends"] = [int(v) for v in const[n.input[2]]]
            extra["axes"] = ([int(v) for v in const[n.input[3]]]
                             if len(n.input) > 3 and n.input[3] else None)
            extra["steps"] = ([int(v) for v in const[n.input[4]]]
                              if len(n.input) > 4 and n.input[4] else None)
            ins = [buffers[n.input[0]]]
        elif op == "Pad":
            extra["pads"] = [int(v) for v in const[n.input[1]]]
            extra["mode"] = attrs.get("mode", "constant")
            ins = [buffers[n.input[0]]]
        elif op == "Transpose":
            extra["perm"] = attrs.get("perm")
            ins = [buffers[n.input[0]]]
        elif op == "Conv":
            for i in n.input:
                if i in const:
                    ensure_weight(i)
            extra["strides"] = attrs.get("strides")
            extra["pads"] = attrs.get("pads")
            extra["dilations"] = attrs.get("dilations")
            extra["group"] = attrs.get("group", 1)
            ins = [buffers[i] if i not in const else None for i in n.input]
        elif op == "ReduceSum":
            if len(n.input) > 1 and n.input[1]:
                extra["axes"] = [int(v) for v in const[n.input[1]]]
            else:
                extra["axes"] = attrs.get("axes")
            extra["keepdims"] = attrs.get("keepdims", 1)
            ins = [buffers[n.input[0]]]
        elif op in ("ReduceMin", "ReduceMax"):
            extra["axes"] = attrs.get("axes")
            extra["keepdims"] = attrs.get("keepdims", 1)
            ins = [buffers[n.input[0]]]
        elif op == "Cast":
            extra["to"] = attrs["to"]
            ins = [buffers[n.input[0]]]
        else:
            # 通用多输入（含广播常量）；保留属性（如 Concat 的 axis）
            extra = dict(attrs)
            for i in n.input:
                if i == "":
                    ins.append(None)
                elif i in const:
                    ensure_weight(i)
                    ins.append(None)
                else:
                    ins.append(buffers[i])

        # 记录常量输入引用
        const_refs = {}
        for pos, name in enumerate(n.input):
            if name and name in const:
                const_refs[str(pos)] = name

        ir_nodes.append({
            "op": op,
            "ins": ins,
            "outs": outs,
            "attrs": extra,
            "constRefs": const_refs,
            "outShapes": [list(shapes[o]) for o in n.output if o],
        })

    # ---- 4) 输出 ----
    graph_outputs = []
    for o in g.output:
        graph_outputs.append({"name": o.name, "buffer": buffers[o.name],
                              "shape": [int(d) for d in shapes[o.name]]})

    # ---- 5) 权重名字表 ----
    ir = {
        "input": {"name": iname, "buffer": buffers[iname], "shape": list(shapes[iname])},
        "outputs": graph_outputs,
        "nodes": ir_nodes,
        "weights": wtensors,
        "nBuffers": next_buf,
        "bufferShapes": {str(k): v for k, v in buf_shapes.items()},
    }
    with open(os.path.join(OUT_DIR, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=1)
    with open(os.path.join(OUT_DIR, "weights.bin"), "wb") as f:
        f.write(bytes(wblob))

    # ---- 6) 验证夹具 ----
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    ort_names = [o.name for o in sess.get_outputs()]
    ort_vals = sess.run(ort_names, {iname: audio})
    with open(os.path.join(OUT_DIR, "test_input.bin"), "wb") as f:
        f.write(audio.astype(np.float32).tobytes())
    fixture = {"input": audio.astype(np.float32).tobytes().__len__() // 4, "outputs": []}
    for name, val in zip(ort_names, ort_vals):
        fn = "test_out_%s.bin" % name.replace(":", "_")
        with open(os.path.join(OUT_DIR, fn), "wb") as f:
            f.write(np.ascontiguousarray(val, dtype=np.float32).tobytes())
        fixture["outputs"].append({"name": name, "file": fn, "shape": list(val.shape)})
    with open(os.path.join(OUT_DIR, "fixture.json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)

    tot = sum(int(np.prod(v)) for v in buf_shapes.values())
    print("缓冲区数: %d, 元素总数: %d (%.2f MB float32)" % (next_buf, tot, tot * 4 / 1024 / 1024))
    print("权重 blob: %.2f KB, 常量张量数: %d" % (len(wblob) / 1024, len(wtensors)))
    print("产物已写入:", OUT_DIR)


if __name__ == "__main__":
    main()