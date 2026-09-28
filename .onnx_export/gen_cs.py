"""从导出的 IR 生成纯 C# 前向引擎所需的图描述数据（NmpIr.g.cs）。

产物：BasicPitch/NmpIr.g.cs
  - 缓冲区元数据（元素数 / 形状）
  - 节点表（算子 / 输入 / 输出 / 属性）
  - 权重常量张量（base64 内联，运行时解码为 float[]）

这样 C# 侧无需任何 JSON 解析或外部依赖，桌面/Unity 通用。
"""
import base64
import json
import os

import numpy as np

OUT_DIR = r"d:\MusicRecorder\.onnx_export\out"
CS_PATH = r"d:\MusicRecorder\BasicPitch\NmpIr.g.cs"

# 算子编码（必须与 C# 侧 NmpIr 常量一致）
OPS = ["Reshape", "Slice", "Pad", "Unsqueeze", "Conv", "Neg", "Transpose", "Concat",
       "Mul", "ReduceSum", "Sqrt", "Add", "Log", "ReduceMin", "Sub", "ReduceMax",
       "Div", "Equal", "Where", "Relu", "Sigmoid"]
OPC = {o: i for i, o in enumerate(OPS)}
PAD_MODE = {"constant": 0, "reflect": 1, "edge": 2}


def emit_ints(name, vals, per_line=20):
    out = ["    public static readonly int[] %s = new int[%d]" % (name, len(vals))]
    out[0] += "\n    {\n"
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(str(int(v)) for v in vals[i:i + per_line])
        out.append("        " + chunk + ",\n")
    out.append("    };\n")
    return "".join(out)


def emit_bytes(name, vals, per_line=30):
    out = ["    public static readonly byte[] %s = new byte[%d]" % (name, len(vals))]
    out[0] += "\n    {\n"
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(str(int(v)) for v in vals[i:i + per_line])
        out.append("        " + chunk + ",\n")
    out.append("    };\n")
    return "".join(out)


def prod(shape):
    p = 1
    for x in shape:
        p *= int(x)
    return p


def main():
    ir = json.load(open(os.path.join(OUT_DIR, "ir.json"), encoding="utf-8"))
    wbin = open(os.path.join(OUT_DIR, "weights.bin"), "rb").read()
    wflat = np.frombuffer(wbin, dtype=np.float32)
    winfo = ir["weights"]

    nbuffers = ir["nBuffers"]
    bshape = {int(k): [int(x) for x in v] for k, v in ir["bufferShapes"].items()}

    # ---- 权重：只打包被保留节点引用的常量张量，按名字排序保证确定性 ----
    used_names = sorted({nm for n in ir["nodes"] for nm in n["constRefs"].values()
                         if nm in winfo})
    widx = {nm: i for i, nm in enumerate(used_names)}
    w_tensors = []
    for nm in used_names:
        info = winfo[nm]
        off = info["offset"] // 4
        cnt = info["count"]
        arr = np.ascontiguousarray(wflat[off:off + cnt], dtype=np.float32)
        assert arr.size == cnt, nm
        w_tensors.append((arr, [int(s) for s in info["shape"]]))

    wdata = np.concatenate([a for a, _ in w_tensors]) if w_tensors else np.zeros(0, np.float32)
    w_off, w_cnt, w_rank, w_shpdata, w_shapeoff = [], [], [], [], []
    cursor = 0
    for arr, shp in w_tensors:
        w_off.append(cursor)
        cursor += arr.size
    for arr, shp in w_tensors:
        w_cnt.append(int(arr.size))
        w_rank.append(len(shp))
        w_shapeoff.append(len(w_shpdata))
        w_shpdata.extend(shp)
    w_shapeoff.append(len(w_shpdata))

    # ---- 节点表 ----
    op_arr, in0, in1, in2, out_arr = [], [], [], [], []
    list_data, list_off = [], [0]

    for n in ir["nodes"]:
        op = n["op"]
        assert op in OPC, op
        op_arr.append(OPC[op])
        out_arr.append(int(n["outs"][0]))

        ins = list(n["ins"]) + [None] * (3 - len(n["ins"]))
        enc = []
        for pos in range(3):
            nm = n.get("constRefs", {}).get(str(pos))
            if nm is not None and nm in widx:
                enc.append(-widx[nm] - 2)
            elif ins[pos] is None:
                enc.append(-1)
            else:
                enc.append(int(ins[pos]))
        in0.append(enc[0])
        in1.append(enc[1])
        in2.append(enc[2])

        attrs = n["attrs"]
        in_shape = bshape[enc[0]] if enc[0] >= 0 else None
        out_shape = n["outShapes"][0]
        a = []
        if op in ("Reshape", "Unsqueeze", "Neg", "Mul", "Add", "Sub", "Div",
                  "Equal", "Relu", "Sigmoid", "Sqrt", "Log", "Where"):
            pass
        elif op == "Slice":
            axes = attrs["axes"]
            starts = attrs["starts"]
            ends = attrs["ends"]
            steps = attrs["steps"] or [1] * len(axes)
            a.append(len(axes))
            r = len(in_shape)
            for k, ax in enumerate(axes):
                ax = int(ax) % r
                dim = in_shape[ax]
                s = int(starts[k])
                e = int(ends[k])
                st = int(steps[k])
                if s < 0:
                    s += dim
                if e < 0:
                    e += dim
                e = min(e, dim)
                s = max(0, s)
                a += [ax, s, e, st]
        elif op == "Pad":
            r = len(in_shape)
            pads = [int(v) for v in attrs["pads"]]
            before = pads[:r]
            after = pads[r:]
            a.append(r)
            a += before
            a += after
            a.append(PAD_MODE.get(attrs.get("mode", "constant"), 0))
        elif op == "Transpose":
            perm = [int(p) for p in attrs["perm"]]
            a.append(len(perm))
            a += perm
        elif op == "Concat":
            ax = int(attrs["axis"]) % len(out_shape)
            bufs = [int(x) for x in n["ins"]]
            a.append(ax)
            a.append(len(bufs))
            a += bufs
        elif op == "Conv":
            r = len(in_shape) - 2
            strides = attrs.get("strides") or [1] * r
            pads = attrs.get("pads") or [0] * (2 * r)
            dil = attrs.get("dilations") or [1] * r
            group = int(attrs.get("group", 1))
            a.append(r)
            a += [int(v) for v in strides]
            a += [int(v) for v in pads[:r]]
            a += [int(v) for v in pads[r:]]
            a += [int(v) for v in dil]
            a.append(group)
        elif op in ("ReduceSum", "ReduceMin", "ReduceMax"):
            keepdims = int(attrs.get("keepdims", 1))
            axes = attrs.get("axes")
            if axes is None:
                axes = list(range(len(in_shape)))
            r = len(in_shape)
            axes = [int(x) % r for x in axes]
            a.append(keepdims)
            a.append(len(axes))
            a += axes
        else:
            raise NotImplementedError(op)

        list_data += a
        list_off.append(len(list_data))

    # ---- 缓冲区元数据 ----
    buf_count, buf_rank, shape_data, shape_off = [], [], [], []
    for b in range(nbuffers):
        shp = bshape[b]
        buf_count.append(prod(shp))
        buf_rank.append(len(shp))
        shape_off.append(len(shape_data))
        shape_data.extend(shp)
    shape_off.append(len(shape_data))

    # ---- 输出缓冲区（按 ONNX 输出名排序 :0/:1/:2）----
    outs = sorted(ir["outputs"], key=lambda o: o["name"])
    assert len(outs) == 3
    buf_c = outs[0]["buffer"]   # :0 contour
    buf_n = outs[1]["buffer"]   # :1 note
    buf_o = outs[2]["buffer"]   # :2 onset

    src = []
    src.append("// <auto-generated>\n")
    src.append("// 由 .onnx_export/gen_cs.py 从 BasicPitch/nmp.onnx 生成，请勿手工修改。\n")
    src.append("// 内容：常量折叠后的前向图描述 + 权重常量。\n")
    src.append("using System;\n\n")
    src.append("namespace BasicPitch;\n\n")
    src.append("/// <summary>\n")
    src.append("/// Basic Pitch (nmp) 常量折叠后的前向图描述 + 权重常量（由 nmp.onnx 生成）。\n")
    src.append("/// 桌面引擎（NmpEngine）与 Unity Burst 引擎（NmpBurstEngine）共用同一份数据。\n")
    src.append("/// </summary>\n")
    src.append("public static class NmpIr\n{\n")
    src.append("    // ---- 算子编码 ----\n")
    for i, o in enumerate(OPS):
        src.append("    public const byte %s = %d;\n" % (o, i))
    src.append("\n    public const int NBuffers = %d;\n" % nbuffers)
    src.append("    public const int NNodes = %d;\n" % len(op_arr))
    src.append("    public const int NWeights = %d;\n\n" % len(w_tensors))
    src.append("    // ---- 图输入 / 输出缓冲区 ----\n")
    src.append("    public const int InputBuffer = %d;\n" % ir["input"]["buffer"])
    src.append("    public const int BufferContour = %d;  // StatefulPartitionedCall:0\n" % buf_c)
    src.append("    public const int BufferNote = %d;     // StatefulPartitionedCall:1\n" % buf_n)
    src.append("    public const int BufferOnset = %d;    // StatefulPartitionedCall:2\n\n" % buf_o)

    src.append("    // ---- 缓冲区元数据（元素数 / 形状）----\n")
    src.append(emit_ints("BufferCount", buf_count))
    src.append(emit_ints("BufferRank", buf_rank))
    src.append(emit_ints("ShapeOff", shape_off))
    src.append(emit_ints("ShapeData", shape_data))
    src.append("\n    // ---- 节点表 ----\n")
    src.append("    // 输入编码：>=0 缓冲区索引；-1 缺省；<=-2 权重索引 = (-v-2)\n")
    src.append(emit_bytes("Op", op_arr))
    src.append(emit_ints("In0", in0))
    src.append(emit_ints("In1", in1))
    src.append(emit_ints("In2", in2))
    src.append(emit_ints("Out", out_arr))
    src.append("    // 变长属性：节点 i 的属性 = List[ListOff[i] .. ListOff[i+1]]\n")
    src.append(emit_ints("ListOff", list_off))
    src.append(emit_ints("List", list_data))
    src.append("\n    // ---- 权重常量张量 ----\n")
    src.append(emit_ints("WeightOff", w_off))
    src.append(emit_ints("WeightCount", w_cnt))
    src.append(emit_ints("WeightRank", w_rank))
    src.append(emit_ints("WeightShapeOff", w_shapeoff))
    src.append(emit_ints("WeightShapeData", w_shpdata))
    b64 = base64.b64encode(np.ascontiguousarray(wdata, dtype=np.float32).tobytes()).decode()
    src.append("    private const string WeightB64 =\n")
    for i in range(0, len(b64), 100):
        src.append('        "%s" +\n' % b64[i:i + 100])
    src.append('        "";\n\n')
    src.append("    /// <summary>所有常量张量的 float32 数据（按 WeightOff 索引）。</summary>\n")
    src.append("    public static readonly float[] WeightData = DecodeWeights();\n\n")
    src.append("    private static float[] DecodeWeights()\n    {\n")
    src.append("        var bytes = Convert.FromBase64String(WeightB64);\n")
    src.append("        var f = new float[bytes.Length / 4];\n")
    src.append("        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);\n")
    src.append("        return f;\n    }\n")
    src.append("}\n")

    with open(CS_PATH, "w", encoding="utf-8") as f:
        f.write("".join(src))

    print("节点数:", len(op_arr), " 缓冲区:", nbuffers)
    print("权重张量:", len(w_tensors), " 元素:", int(wdata.size),
          " (%.1f KB)" % (wdata.size * 4 / 1024))
    print("base64 长度:", len(b64))
    print("已写入:", CS_PATH)


if __name__ == "__main__":
    main()