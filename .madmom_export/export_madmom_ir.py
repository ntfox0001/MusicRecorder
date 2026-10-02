# -*- coding: utf-8 -*-
"""将 madmom 的 pickled NeuralNetwork（downbeat / beat 模型）导出为与 BasicPitch 同构的
常量折叠 IR + 内联权重，供 C# 的 MadmomEngine 解释执行。

与 BasicPitch 的 export_ir.py 区别：madmom 模型不是 ONNX，而是
`madmom.ml.nn.NeuralNetwork`（内部是 LSTMLayer / GRULayer / BidirectionalLayer /
FeedForwardLayer 组成的层栈）。本脚本直接走它的权重组，按 MadmomEngine 期望的
门序 / 权重布局重排，得到一份可直接喂给 gen_madmom_cs.py 的 ir.json。

权重命名约定（必须与 MadmomEngine 完全一致）：
  - BLSTM 节点 idx（双向）：
        lstm_{idx}fx / lstm_{idx}fh / lstm_{idx}fb   （前向）
        lstm_{idx}bx / lstm_{idx}bh / lstm_{idx}bb   （后向）
    GRU 双向把前缀换成 gru_。
  - 单向 LSTM/GRU 节点 idx：
        lstm_{idx}x / lstm_{idx}h / lstm_{idx}b  （GRU 用 gru_ 前缀）
  - Dense 节点 idx：
        dense{idx}_x (out×in) / dense{idx}_b (out)

门序 / 布局（依据 madmom 源码 madmom/ml/nn/layers.py）：
  - LSTM 门序 [input, forget, cell, output]；每层 4 个独立 Gate，各自
        weights=(in,H)   recurrent=(H,H)   bias=(H,)
    → Wx = concat([Wi.T, Wf.T, Wc.T, Wo.T], axis=0)  (4H, in)
      Wh = concat([Ui,  Uf,  Uc,  Uo ], axis=0)  (4H, H)
      b  = concat([bi,  bf,  bc,  bo ], axis=0)  (4H,)
  - GRU 门序 [reset, update, cell]；候选 = tanh(Wc·x + b + r ⊙ (Uc·h))，输出 = u·cand + (1-u)·h
    引擎块序为 [z(update), r(reset), n(candidate)]，故重排为 [Wu, Wr, Wc]：
      Wx = concat([Wu.T, Wr.T, Wc.T], axis=0)  (3H, in)
      Wh = concat([Uu,   Ur,   Uc ], axis=0)  (3H, H)
      b  = concat([bu,   br,   bc ], axis=0)  (3H,)
  - BLSTM 输出 np.hstack([fwd, bwd[::-1]])，即特征轴 [前向;后向]，与引擎一致。
  - Dense: out = x·W(in,out) + b → 引擎用 (out,in)，故存 W.T。

产物（写到 OUT_DIR）：
  ir.json         拓扑 + 缓冲区形状 + 命名权重(base64) + meta
  madmom_weights.bin  预留（CNN 常量，RNN 模型通常为空）

用法：
  python export_madmom_ir.py --model <madmom .pkl> [--fps 100] [--out OUT_DIR]
  python export_madmom_ir.py --model <madmom .pkl> --verify    # 数值自检
"""
import argparse
import base64
import json
import os

import numpy as np

# 算子编码必须与 C# 侧 IrGraph 常量一致（0..25）
OPS = ["Reshape", "Slice", "Pad", "Unsqueeze", "Conv", "Neg", "Transpose",
       "Concat", "Mul", "ReduceSum", "Sqrt", "Add", "Log", "ReduceMin",
       "Sub", "ReduceMax", "Div", "Equal", "Where", "Relu", "Sigmoid",
       "Lstm", "Gru", "Blstm", "Dense", "Tanh", "Softmax"]
OPC = {o: i for i, o in enumerate(OPS)}

# madmom 激活函数名 → 算子
ACT_OP = {"sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "Relu", "softmax": "Softmax"}


# ----------------------------------------------------------------------------- 载入
def load_nn(path):
    """载入单个 madmom NeuralNetwork .pkl，无需编译 madmom 的 C 扩展。

    做法：把顶层 `madmom` 与 `madmom.ml` 包 stub 掉（仅设置 __path__ 指向纯 Python
    源码目录），这样导入 madmom.ml.nn / madmom.ml.nn.layers 时不会触发 madmom/__init__.py
    （依赖 audio C 扩展与 PyPI 包元数据），也不会触发 madmom.ml/__init__.py 里的
    hmm/gmm/crf 重依赖。权重 .pkl 来自 PyPI sdist（与源码同版本）。
    """
    import sys, types, pickle
    SRC = os.environ.get(
        "MADMOM_SRC",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     ".madmom_pkg", "madmom-0.16.1", "madmom"))
    if not os.path.isdir(SRC):
        raise RuntimeError("找不到 madmom 纯 Python 源码目录: %s（请用 MADMOM_SRC 指定）" % SRC)

    # Python 3.10+ 把 collections.abc 的名称移出 collections；madmom 0.16.1 仍从 collections 导入
    import collections, collections.abc
    for _n in ("MutableSequence", "MutableMapping", "MutableSet", "Sequence",
               "Mapping", "Set", "Iterable", "Iterator", "Callable", "Hashable",
               "ABCMeta"):
        if not hasattr(collections, _n) and hasattr(collections.abc, _n):
            setattr(collections, _n, getattr(collections.abc, _n))

    if "madmom" not in sys.modules:
        m = types.ModuleType("madmom")
        m.__path__ = [SRC]
        sys.modules["madmom"] = m
    # stub madmom.ml：跳过 hmm/gmm/crf 等重依赖（仅需要 madmom.ml.nn）
    if "madmom.ml" not in sys.modules:
        ml = types.ModuleType("madmom.ml")
        ml.__path__ = [os.path.join(SRC, "ml")]
        sys.modules["madmom.ml"] = ml
    # Python 3.12+ 移除了 distutils；madmom.ml.nn.activations 用到 distutils.version.LooseVersion
    if "distutils" not in sys.modules:
        _du = types.ModuleType("distutils")
        _duv = types.ModuleType("distutils.version")

        class _LooseVersion:
            def __init__(self, v):
                self.v = str(v)

            def _cmp(self, o):
                import re
                a = [int(x) for x in re.findall(r"\d+", self.v)]
                b = [int(x) for x in re.findall(r"\d+", str(o))]
                return (a > b) - (a < b)

            def __lt__(self, o):
                return self._cmp(o) < 0

            def __gt__(self, o):
                return self._cmp(o) > 0

            def __eq__(self, o):
                return self._cmp(o) == 0

        _duv.LooseVersion = _LooseVersion
        _du.version = _duv
        sys.modules["distutils"] = _du
        sys.modules["distutils.version"] = _duv
    from madmom.ml.nn import NeuralNetwork
    with open(path, "rb") as f:
        # madmom 的 .pkl 多为旧协议（含 py2 字节串），用 latin-1 1:1 解码避免 ascii 报错
        nn = pickle.load(f, encoding="latin-1")
    if not isinstance(nn, NeuralNetwork):
        raise RuntimeError("仅支持单个 NeuralNetwork（非 ensemble）；请直接传单个 .pkl")
    return nn


def _gate_w(gate):
    """取一个 Gate/GRUCell 的 (W_in(in,H), U(H,H), b(H,))，转成连续 float32。"""
    W = np.asarray(gate.weights, dtype=np.float32)          # (in, H)
    U = np.asarray(gate.recurrent_weights, dtype=np.float32)  # (H, H)
    b = np.asarray(gate.bias, dtype=np.float32).reshape(-1)   # (H,)
    return W, U, b


def collect_rnn(rnn_layer, pfx, node, direction):
    """收集单方向（前向/后向/单向）RNN 的权重，命名并加入 named。
    返回 (H, ) 并把权重写入 named 字典。direction: '' / 'f' / 'b'。
    """
    named = {}
    has = lambda a: hasattr(rnn_layer, a)

    def _pe(gate, H):
        pw = getattr(gate, "peephole_weights", None)
        if pw is None:
            return np.zeros(H, np.float32)
        return np.asarray(pw, dtype=np.float32).reshape(-1)

    if has("input_gate"):  # LSTM
        ig, fg, cg, og = (rnn_layer.input_gate, rnn_layer.forget_gate,
                           rnn_layer.cell, rnn_layer.output_gate)
        Wi, Ui, bi = _gate_w(ig)
        Wf, Uf, bf = _gate_w(fg)
        Wc, Uc, bc = _gate_w(cg)
        Wo, Uo, bo = _gate_w(og)
        H = Wi.shape[1]
        Wx = np.concatenate([Wi.T, Wf.T, Wc.T, Wo.T], axis=0)   # (4H, in)
        # 循环权重必须转置：madmom 用 prev@U（列索引），引擎按行索引 Σ_j W[k,j]h[j]
        # 需要 W[k,j]=U[j,k]，即存 U.T。
        Wh = np.concatenate([Ui.T, Uf.T, Uc.T, Uo.T], axis=0)   # (4H, H)
        b = np.concatenate([bi, bf, bc, bo], axis=0)            # (4H,)
        # 窥孔权重：input/forget/output 有，cell 无 → cell 槽填 0
        pe = np.concatenate([_pe(ig, H), _pe(fg, H), np.zeros(H, np.float32), _pe(og, H)])
        kind = "lstm"
    elif has("reset_gate"):  # GRU
        rg, ug, cg = (rnn_layer.reset_gate, rnn_layer.update_gate, rnn_layer.cell)
        Wr, Ur, br = _gate_w(rg)
        Wu, Uu, bu = _gate_w(ug)
        Wc, Uc, bc = _gate_w(cg)
        H = Wr.shape[1]
        # 引擎块序 [z(update), r(reset), n(candidate)] → 重排 [Wu, Wr, Wc]
        Wx = np.concatenate([Wu.T, Wr.T, Wc.T], axis=0)   # (3H, in)
        # 循环权重转置（同 LSTM 说明）
        Wh = np.concatenate([Uu.T, Ur.T, Uc.T], axis=0)   # (3H, H)
        b = np.concatenate([bu, br, bc], axis=0)          # (3H,)
        pe = np.zeros(3 * H, np.float32)                  # GRU 无窥孔
        kind = "gru"
    else:
        raise RuntimeError("未知 RNN 层类型")

    assert Wx.shape[0] == (4 * H if kind == "lstm" else 3 * H)
    assert Wh.shape == (Wx.shape[0], H)
    assert b.shape[0] == Wx.shape[0]
    assert pe.shape[0] == Wx.shape[0]

    base = "%s_%s%s" % (kind, node, direction)  # e.g. lstm_3f
    named[base + "x"] = Wx
    named[base + "h"] = Wh
    named[base + "b"] = b
    named[base + "p"] = pe
    return H, named


def is_bidirectional(layer):
    return hasattr(layer, "fwd_layer") and hasattr(layer, "bwd_layer")


def extract(nn):
    """遍历 nn.layers，构建 IR。返回 (nodes, buffer_shapes, named, meta)。"""
    nodes = []
    buf_shapes = {}        # buffer index -> list[int]
    named = {}
    next_buf = 1          # 0 保留给输入
    input_buf = 0
    input_feature_dim = None

    def new_buf(shape):
        nonlocal next_buf
        buf_shapes[next_buf] = list(int(d) for d in shape)
        next_buf += 1
        return next_buf - 1

    # 输入形状在导出时未知（依赖运行时 T）；用（T=1, F）占位，引擎只关心最后一维
    buf_shapes[input_buf] = [1, 1]  # 占位，后续由第一层权重推断 F

    def emit(op, in0, in1=-1, in2=-1, out_shape=None, lst=None):
        idx = len(nodes)
        ob = new_buf(out_shape if out_shape else [1])
        nodes.append({
            "op": op,
            "ins": [in0, in1, in2],
            "outs": [ob],
            "list": lst if lst is not None else [],
            "outShape": list(out_shape) if out_shape else [1],
        })
        return idx, ob

    last_out = input_buf
    for layer in nn.layers:
        cls = type(layer).__name__

        if cls == "Dropout":
            # 推理期恒等
            continue

        if cls in ("LSTMLayer", "GRULayer"):
            is_lstm = cls == "LSTMLayer"
            kind = "lstm" if is_lstm else "gru"
            g0 = layer.input_gate if hasattr(layer, "input_gate") else layer.reset_gate
            H = int(_gate_w(g0)[0].shape[1])
            # 单向 RNN 输出 (T, H)
            idx, ob = emit("Lstm" if is_lstm else "Gru", last_out,
                           out_shape=[1, H], lst=[H, 0, 1])
            _, w = collect_rnn(layer, kind, idx, "")
            named.update(w)
            buf_shapes[ob] = [1, H]
            last_out = ob

        elif is_bidirectional(layer):
            fwd = layer.fwd_layer
            bwd = layer.bwd_layer
            fkind = "lstm" if hasattr(fwd, "input_gate") else "gru"
            bkind = "lstm" if hasattr(bwd, "input_gate") else "gru"
            assert fkind == bkind, "双向前后向类型须一致"
            fg0 = fwd.input_gate if hasattr(fwd, "input_gate") else fwd.reset_gate
            bg0 = bwd.input_gate if hasattr(bwd, "input_gate") else bwd.reset_gate
            Hf = int(_gate_w(fg0)[0].shape[1])
            Hb = int(_gate_w(bg0)[0].shape[1])
            assert Hf == Hb
            idx, ob = emit("Blstm", last_out, out_shape=[1, 2 * Hf], lst=[Hf, 1, 1])
            _, wf = collect_rnn(fwd, fkind, idx, "f")
            _, wb = collect_rnn(bwd, bkind, idx, "b")
            named.update(wf); named.update(wb)
            buf_shapes[ob] = [1, 2 * Hf]
            last_out = ob

        elif cls == "FeedForwardLayer":
            W = np.asarray(layer.weights, dtype=np.float32)   # (in, out)
            b = np.asarray(layer.bias, dtype=np.float32).reshape(-1)  # (out,)
            out_dim = W.shape[1]
            idx, ob = emit("Dense", last_out, out_shape=[1, out_dim], lst=[])
            named["dense%d_x" % idx] = W.T.copy()   # (out, in)
            named["dense%d_b" % idx] = b
            buf_shapes[ob] = [1, out_dim]
            last_out = ob
            # 激活（若有）作为独立节点
            afn = getattr(layer, "activation_fn", None)
            if afn is not None:
                aname = getattr(afn, "__name__", None)
                if aname in ACT_OP:
                    aop = ACT_OP[aname]
                    idx2, ob2 = emit(aop, last_out, out_shape=[1, out_dim], lst=[])
                    last_out = ob2

        elif cls == "Activation":
            afn = getattr(layer, "activation_fn", None)
            aname = getattr(afn, "__name__", None) if afn else None
            if aname in ACT_OP:
                out_shape = buf_shapes[last_out]
                idx2, ob2 = emit(ACT_OP[aname], last_out, out_shape=out_shape, lst=[])
                last_out = ob2
            else:
                # 不支持的激活（如 softmax）暂不处理
                raise RuntimeError("不支持的激活: %s" % aname)

        else:
            raise RuntimeError("暂不支持的层类型: %s" % cls)

        # 记录输入特征维度（第一层权重决定）
        if input_feature_dim is None:
            if cls in ("LSTMLayer", "GRULayer") or is_bidirectional(layer):
                # 取前向第一层的输入权重 in 维
                if is_bidirectional(layer):
                    g0 = layer.fwd_layer.input_gate if hasattr(layer.fwd_layer, "input_gate") else layer.fwd_layer.reset_gate
                else:
                    g0 = layer.input_gate if hasattr(layer, "input_gate") else layer.reset_gate
                input_feature_dim = int(_gate_w(g0)[0].shape[0])
            elif cls == "FeedForwardLayer":
                input_feature_dim = int(np.asarray(layer.weights).shape[0])

    output_buf = last_out
    output_dim = buf_shapes[output_buf][-1]
    # 输入缓冲区最后一维 = 输入特征维（引擎据此由输入长度推导 T = len/F）
    if input_feature_dim is not None:
        buf_shapes[input_buf] = [1, input_feature_dim]
    meta = {
        "inputBuffer": input_buf,
        "outputBuffer": output_buf,
        "inputFeatureDim": input_feature_dim,
        "outputDim": output_dim,
        "nBuffers": next_buf,
    }
    return nodes, buf_shapes, named, meta


def build_ir_dict(nodes, buf_shapes, named, meta, fps):
    # 命名权重 base64
    named_b64 = {}
    for name, arr in named.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        named_b64[name] = {
            "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape),
        }
    return {
        "meta": dict(meta, fps=fps),
        "nodes": nodes,
        "bufferShapes": {str(k): v for k, v in buf_shapes.items()},
        "namedWeights": named_b64,
    }


# -------------------------------------------------------------------- numpy 复刻（自检用）
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    # 与 madmom 一致：沿最后一维做稳定 softmax（先减行最大值）
    tmp = np.max(x, axis=-1, keepdims=True)
    out = np.exp(x - tmp)
    out /= np.sum(out, axis=-1, keepdims=True)
    return out


def _step_rnn(gru, Wx, Wh, b, pe, x_row, h, c, out_row):
    # 与 C# 引擎完全一致：权重按 C 连续扁平化为 1D，门 k 的输入权重落在
    # [k*F, k*F+F)，循环权重落在 [k*H, k*H+H)；GRU 候选项 Wh[(2H+j)*H+k2]。
    # LSTM 窥孔：ig/fg 用上一时刻 cell 状态，og 用本时刻（更新后）cell 状态。
    Wx = np.ascontiguousarray(Wx, dtype=np.float32).ravel()
    Wh = np.ascontiguousarray(Wh, dtype=np.float32).ravel()
    pe = np.ascontiguousarray(pe, dtype=np.float32).ravel() if pe is not None else None
    gates = 3 if gru else 4
    H = h.shape[0]
    F = x_row.shape[0]
    pre = b.copy()
    for k in range(gates * H):
        s = b[k]
        baseK = k * F
        s += Wx[baseK:baseK + F] @ x_row
        baseH = k * H
        s += Wh[baseH:baseH + H] @ h
        pre[k] = s
    if gru:
        z = _sigmoid(pre[0:H])
        r = _sigmoid(pre[H:2 * H])
        n_pre = pre[2 * H:3 * H].copy()
        for j in range(H):
            base = (2 * H + j) * H
            n_pre[j] += r[j] * (Wh[base:base + H] @ h)
        nn = np.tanh(n_pre)
        out_row[:] = (1 - z) * nn + z * h
        h[:] = out_row
    else:
        ci = _sigmoid(pre[0:H] + (pe[0:H] * c if pe is not None else 0))
        co = _sigmoid(pre[H:2 * H] + (pe[H:2 * H] * c if pe is not None else 0))
        c_in = np.tanh(pre[2 * H:3 * H])
        c[:] = co * c + ci * c_in
        o = _sigmoid(pre[3 * H:4 * H] + (pe[3 * H:4 * H] * c if pe is not None else 0))
        out_row[:] = o * np.tanh(c)
        h[:] = out_row


class MadmomInterpreter:
    """纯 numpy 复刻 MadmomEngine，用于 --verify 对比 madmom 原模型。"""
    def __init__(self, ir, named):
        self.nodes = ir["nodes"]
        self.buf_shapes = {int(k): v for k, v in ir["bufferShapes"].items()}
        self.named = named
        self.buffers = {}
        self.meta = ir["meta"]

    def _rnn_weights(self, pfx, node, direction):
        base = "%s_%s%s" % (pfx, node, direction)
        return (self.named[base + "x"], self.named[base + "h"],
                self.named[base + "b"], self.named.get(base + "p"))

    def run(self, x):
        T, F = x.shape
        inb = self.meta["inputBuffer"]
        self.buffers = {inb: x.astype(np.float32)}
        for n in self.nodes:
            op = n["op"]
            ins = n["ins"]
            ob = n["outs"][0]
            if op in ("Lstm", "Gru", "Blstm"):
                xi = self.buffers[ins[0]]
                bidir = 1 if op == "Blstm" else 0
                gru = 1 if op == "Gru" else 0
                H = n["list"][0]
                outF = 2 * H if bidir else H
                dst = np.zeros((T, outF), dtype=np.float32)
                pfx = "gru" if gru else "lstm"
                if not bidir:
                    Wx, Wh, b, pe = self._rnn_weights(pfx, self.nodes.index(n), "")
                    h = np.zeros(H, np.float32); c = np.zeros(H, np.float32)
                    row = np.zeros(H, np.float32)
                    for t in range(T):
                        _step_rnn(bool(gru), Wx, Wh, b, pe, xi[t], h, c, row)
                        dst[t] = row
                else:
                    Wxf, Whf, bf, pef = self._rnn_weights(pfx, self.nodes.index(n), "f")
                    Wxb, Whb, bb, peb = self._rnn_weights(pfx, self.nodes.index(n), "b")
                    hf = np.zeros(H, np.float32); cf = np.zeros(H, np.float32)
                    hb = np.zeros(H, np.float32); cb = np.zeros(H, np.float32)
                    rf = np.zeros(H, np.float32); rb = np.zeros(H, np.float32)
                    for t in range(T):
                        srcIdx = T - 1 - t
                        _step_rnn(bool(gru), Wxf, Whf, bf, pef, xi[t], hf, cf, rf)
                        dst[t, 0:H] = rf
                        _step_rnn(bool(gru), Wxb, Whb, bb, peb, xi[srcIdx], hb, cb, rb)
                        dst[srcIdx, H:2 * H] = rb
                self.buffers[ob] = dst
            elif op == "Dense":
                xi = self.buffers[ins[0]]
                W = self.named["dense%d_x" % self.nodes.index(n)]
                b = self.named["dense%d_b" % self.nodes.index(n)]
                self.buffers[ob] = xi @ W.T + b
            elif op == "Sigmoid":
                self.buffers[ob] = _sigmoid(self.buffers[ins[0]])
            elif op == "Tanh":
                self.buffers[ob] = np.tanh(self.buffers[ins[0]])
            elif op == "Softmax":
                self.buffers[ob] = _softmax(self.buffers[ins[0]])
            elif op == "Relu":
                self.buffers[ob] = np.maximum(0, self.buffers[ins[0]])
            else:
                raise RuntimeError("自检解释器不支持算子: %s" % op)
        return self.buffers[self.meta["outputBuffer"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="madmom .pkl 模型路径")
    ap.add_argument("--fps", type=float, default=100.0, help="特征帧率（downbeat 默认 100）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--verify", action="store_true", help="数值自检：对比 madmom 原模型输出")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nn = load_nn(args.model)
    nodes, buf_shapes, named, meta = extract(nn)
    ir = build_ir_dict(nodes, buf_shapes, named, meta, args.fps)

    with open(os.path.join(args.out, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=1)
    # 预留 CNN 常量 blob（RNN 模型为空）
    with open(os.path.join(args.out, "madmom_weights.bin"), "wb") as f:
        f.write(b"")

    print("节点数:", len(nodes), " 缓冲区:", meta["nBuffers"])
    print("输入特征维:", meta["inputFeatureDim"], " 输出维:", meta["outputDim"], " fps:", args.fps)
    print("命名权重数:", len(named), " 总参数:",
          sum(int(np.prod(v.shape)) for v in named.values()))
    print("产物已写入:", os.path.abspath(args.out))

    if args.verify:
        F = meta["inputFeatureDim"]
        rng = np.random.default_rng(0)
        x = rng.standard_normal((60, F)).astype(np.float32)
        ref = nn.process(x.copy())
        mine = MadmomInterpreter(ir, named).run(x)
        ref = np.asarray(ref, dtype=np.float32)
        if ref.shape != mine.shape:
            # madmom 可能在末维 squeeze；尝试对齐
            if ref.ndim == 1 and mine.shape[1] == 1:
                ref = ref.reshape(-1, 1)
        err = np.max(np.abs(ref - mine))
        print("VERIFY max|ref-mine| =", err)
        if err < 1e-3:
            print("VERIFY PASS ✅ 权重映射与引擎数学一致")
        else:
            print("VERIFY FAIL ❌ 请检查门序/权重布局映射")


if __name__ == "__main__":
    main()
