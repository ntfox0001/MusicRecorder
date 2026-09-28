"""在 numpy 上忠实复现 nmp.onnx 的前向计算，作为 C# 移植的参考实现。

设计要点：
- 逐节点按 ONNX 语义求值（graph.node 已是拓扑序）。
- 同时记录每个张量的形状，便于生成 IR。
- 用 onnxruntime 作为标准答案做数值比对。
"""
import numpy as np
import onnx
from onnx import numpy_helper

MODEL = r"d:\MusicRecorder\BasicPitch\nmp.onnx"

# ---------------------------------------------------------------- helpers

def reflect_pad_1d_axis(arr, axis, before, after):
    """沿指定轴做镜像填充（numpy 'reflect' 语义，不含边界值）。"""
    for _ in range(before):
        # 取 axis 上索引 1 处的切片插到最前
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(1, 2)
        arr = np.concatenate([arr[tuple(idx)], arr], axis=axis)
    for _ in range(after):
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(-2, -1)
        arr = np.concatenate([arr, arr[tuple(idx)]], axis=axis)
    return arr


def onnx_pad(x, pads, mode):
    """pads: (2*rank,)  [b0,b1,...,e0,e1,...]"""
    rank = x.ndim
    begins = pads[:rank]
    ends = pads[rank:]
    pad_width = [(int(begins[i]), int(ends[i])) for i in range(rank)]
    if mode == b"reflect":
        # ONNX reflect 与 numpy reflect 语义一致（镜像但不重复边界值）
        return np.pad(x, pad_width, mode="reflect")
    if mode == b"edge":
        return np.pad(x, pad_width, mode="edge")
    # constant
    return np.pad(x, pad_width, mode="constant", constant_values=0.0)


def conv_nd(x, w, b, strides, pads, dilations, group):
    """通用 N 维卷积（NCHW），group 为 1 时走快速路径。"""
    rank = x.ndim - 2                       # 空间维数
    N, C = x.shape[0], x.shape[1]
    M = w.shape[0]
    kshape = w.shape[2:]
    # 显式 pad
    if pads is None or len(pads) == 0:
        pads = [0] * (2 * rank)
    pad_width = [(0, 0), (0, 0)] + [(int(pads[i]), int(pads[rank + i])) for i in range(rank)]
    xp = np.pad(x, pad_width, mode="constant", constant_values=0.0)

    out_shape_sp = []
    for i in range(rank):
        eff_k = (kshape[i] - 1) * dilations[i] + 1
        out_shape_sp.append((xp.shape[2 + i] - eff_k) // strides[i] + 1)

    xp = np.ascontiguousarray(xp, dtype=np.float32)
    out = np.zeros([N, M] + out_shape_sp, dtype=np.float32)

    # 每个空间维的输出基址 / 卷积核偏移
    base_list, off_list = [], []
    for i in range(rank):
        base_list.append(np.arange(out_shape_sp[i]) * strides[i])
        off_list.append(np.arange(kshape[i]) * dilations[i])

    Cg = C // group
    Mg = M // group
    prodK = int(np.prod(kshape))
    prodS = int(np.prod(out_shape_sp))
    for g in range(group):
        patches = xp[:, g * Cg:(g + 1) * Cg]
        for i in range(rank):
            idxs = (base_list[i][:, None] + off_list[i][None, :]).reshape(-1)
            patches = np.take(patches, idxs, axis=2 + i)
        # 形状 [N, Cg, s0*k0, s1*k1, ...] -> [N,Cg, s0,k0, s1,k1,...]
        mid = [N, Cg]
        for i in range(rank):
            mid += [out_shape_sp[i], kshape[i]]
        patches = patches.reshape(mid)
        perm = [0, 1] + [2 + 2 * i + 1 for i in range(rank)] + [2 + 2 * i for i in range(rank)]
        patches = np.transpose(patches, perm).reshape(N, Cg, prodK, prodS)
        wg = w[g * Mg:(g + 1) * Mg]
        wflat = wg.reshape(Mg, Cg, prodK)
        res = np.einsum("mck,nckp->nmp", wflat, patches, optimize=True).astype(np.float32)
        res = res.reshape([N, Mg] + list(out_shape_sp))
        out[:, g * Mg:(g + 1) * Mg] = res

    if b is not None:
        shape = [1, M] + [1] * rank
        out = out + b.reshape(shape)
    return out


# ---------------------------------------------------------------- evaluator

class RefEvaluator:
    def __init__(self, model_path):
        m = onnx.load(model_path)
        g = m.graph
        self.inits = {init.name: numpy_helper.to_array(init) for init in g.initializer}
        self.nodes = list(g.node)
        self.graph_inputs = [i.name for i in g.input if i.name not in self.inits]
        self.graph_outputs = [o.name for o in g.output]
        self.shapes = {}
        self.input_name = self.graph_inputs[0]

    def run(self, feed):
        env = dict(self.inits)
        for name in self.graph_inputs:
            env[name] = feed[name].astype(np.float32)
            self.shapes[name] = tuple(env[name].shape)

        for n in self.nodes:
            op = n.op_type
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

            # 主输入张量（跳过可选缺失输入）
            ins = [env[i] if i != "" else None for i in n.input]
            out = self._eval(op, ins, attrs)
            for oname in n.output:
                if oname == "":
                    continue
                env[oname] = out
                self.shapes[oname] = tuple(np.shape(out))
        return {o: env[o] for o in self.graph_outputs}

    def _eval(self, op, ins, attrs):
        x = ins[0]
        if op == "Reshape":
            return np.reshape(x, ins[1].astype(np.int64)).astype(x.dtype)
        if op == "Slice":
            starts = ins[1].astype(np.int64)
            ends = ins[2].astype(np.int64)
            axes = ins[3].astype(np.int64) if len(ins) > 3 and ins[3] is not None else None
            steps = None
            if len(ins) > 4 and ins[4] is not None:
                steps = ins[4].astype(np.int64)
            sl = [slice(None)] * x.ndim
            dims = range(x.ndim) if axes is None else axes
            for k, ax in enumerate(dims):
                st = int(starts[k])
                en = int(ends[k])
                sp = int(steps[k]) if steps is not None else 1
                sl[int(ax)] = slice(st, en, sp)
            return x[tuple(sl)]
        if op == "Pad":
            return onnx_pad(x, ins[1].astype(np.int64), attrs.get("mode", b"constant"))
        if op == "Unsqueeze":
            axes = ins[1].astype(np.int64) if len(ins) > 1 else np.array(attrs["axes"])
            out = x
            for ax in sorted(int(a) % (x.ndim + len(axes)) for a in axes):
                out = np.expand_dims(out, ax)
            return out
        if op == "Transpose":
            return np.transpose(x, attrs.get("perm", None))
        if op == "Conv":
            weights = ins[1]
            bias = ins[2] if len(ins) > 2 and ins[2] is not None else None
            return conv_nd(x, weights, bias,
                           attrs.get("strides", [1] * (x.ndim - 2)),
                           attrs.get("pads", None),
                           attrs.get("dilations", [1] * (x.ndim - 2)),
                           attrs.get("group", 1))
        if op == "Neg":
            return -x
        if op == "Concat":
            return np.concatenate(ins, axis=attrs["axis"])
        if op == "Mul":
            return x * ins[1]
        if op == "Add":
            return x + ins[1]
        if op == "Sub":
            return x - ins[1]
        if op == "Div":
            return x / ins[1]
        if op == "Sqrt":
            return np.sqrt(x)
        if op == "Log":
            return np.log(x)
        if op == "Relu":
            return np.maximum(x, 0)
        if op == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if op == "Equal":
            return (x == ins[1])
        if op == "Where":
            return np.where(ins[0], ins[1], ins[2])
        if op == "Cast":
            to = attrs["to"]
            npdtype = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}[to]
            return x.astype(npdtype)
        if op == "Shape":
            return np.array(x.shape, dtype=np.int64)
        if op == "ReduceSum":
            axes = ins[1].astype(np.int64) if len(ins) > 1 and ins[1] is not None else None
            if axes is None and attrs.get("noop_with_empty_axes", 0) == 1:
                return x
            return np.sum(x, axis=tuple(axes) if axes is not None else None,
                          keepdims=bool(attrs.get("keepdims", 1)))
        if op in ("ReduceMin", "ReduceMax"):
            axes = attrs.get("axes", None)
            fn = np.min if op == "ReduceMin" else np.max
            return fn(x, axis=tuple(axes) if axes is not None else None,
                      keepdims=bool(attrs.get("keepdims", 1)))
        raise NotImplementedError("未支持的算子: " + op)


if __name__ == "__main__":
    import sys
    import onnxruntime as ort

    ev = RefEvaluator(MODEL)

    rng = np.random.default_rng(1234)
    audio = rng.standard_normal((1, 43844, 1)).astype(np.float32) * 0.3
    feed = {ev.input_name: audio}

    ref = ev.run(feed)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: audio})

    print("ORT 输出名:", [o.name for o in sess.get_outputs()])
    print("ref 输出名:", list(ref.keys()))
    for o in sess.get_outputs():
        r = ref[o.name]
        mx = np.max(np.abs(r))
        print("  %-28s shape=%s  max=%.5f" % (o.name, r.shape, mx))

    for a, b in zip(ort_out, [ref[o.name] for o in sess.get_outputs()]):
        d = np.max(np.abs(a - b))
        print("  diff vs ORT: %.3e   (ort range %.5f..%.5f)" % (d, a.min(), a.max()))