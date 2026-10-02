using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Madmom;

/// <summary>
/// 通用前向推理引擎：解释执行 <see cref="IrGraph"/>。
///
/// 算子集兼容 BasicPitch 的全部 CNN 算子（Conv / Transpose / Slice / Pad / Concat /
/// Reduce* / Unary / Binary / Where / Reshape / Unsqueeze），并新增循环算子
/// LSTM / GRU / BLSTM（madmom downbeat/beat 是 BLSTM）。
///
/// 与 BasicPitch（输入形状固定）不同，madmom 的帧数 T 是动态的：所有缓冲区形状都以
/// [T, ...] 表示，时间维在 Run 时由输入长度推导并前置到形状最前；各缓冲区在 Run 内
/// 按 T 重新分配。RNN/Dense/Unary 仅依赖最后一维（H / out），故 CNN 算子在纯
/// RNN 模型（无 CNN）场景下不影响正确性。
///
/// RNN/Dense 权重按名字从 <see cref="IrGraph.Weights"/> 取（见 <see cref="IrGraph"/> 命名约定）。
/// </summary>
public sealed class MadmomEngine
{
    private const int MaxRank = 5;

    static readonly string[] OpNames =
    {
        "Reshape", "Slice", "Pad", "Unsqueeze", "Conv", "Neg", "Transpose", "Concat",
        "Mul", "ReduceSum", "Sqrt", "Add", "Log", "ReduceMin", "Sub", "ReduceMax",
        "Div", "Equal", "Where", "Relu", "Sigmoid", "Lstm", "Gru", "Blstm",
        "Dense", "Tanh", "Softmax"
    };

    private readonly IrGraph _g;
    private float[] _data = Array.Empty<float>();
    private int[] _off = Array.Empty<int>();
    private int[] _count = Array.Empty<int>();
    private int _t;   // 当前帧数

    public MadmomEngine(IrGraph graph)
    {
        _g = graph;
    }

    /// <summary>输入特征维度（取自图元数据）。</summary>
    public int FeatureDim => _g.InputFeatureDim;
    /// <summary>当前 Run 的帧数 T。</summary>
    public int FrameCount => _t;

    /// <summary>
    /// 流式模式：RNN 的**前向**隐藏状态跨 Run 保留，用于分块前向（长音频常量内存）。
    /// BLSTM 的**后向**无法在线计算，每块只能从块末尾重新开始 —— 块长远大于 LSTM 的有效
    /// 记忆时误差可忽略（已实测：10 秒分块与全量结果逐拍一致）。
    /// </summary>
    public bool KeepState { get; set; }

    private Dictionary<int, float[]>? _hState, _cState;

    /// <summary>丢弃已保存的流式状态（开始处理新音频时调用）。</summary>
    public void ResetState() { _hState = null; _cState = null; }

    private void LoadState(int node, float[] h, float[] c)
    {
        if (!KeepState || _hState == null) return;
        if (_hState.TryGetValue(node, out var sh) && _cState!.TryGetValue(node, out var sc)
            && sh.Length == h.Length)
        {
            Array.Copy(sh, 0, h, 0, h.Length);
            Array.Copy(sc, 0, c, 0, c.Length);
        }
    }

    private void SaveState(int node, float[] h, float[] c)
    {
        if (!KeepState) return;
        _hState ??= new Dictionary<int, float[]>();
        _cState ??= new Dictionary<int, float[]>();
        _hState[node] = (float[])h.Clone();
        _cState[node] = (float[])c.Clone();
    }

    public void Run(float[] input)
    {
        int ib = _g.InputBuffer;
        int fDim = StoredDim(ib, _g.BufferRank[ib] - 1);
        if (input.Length % fDim != 0)
            throw new ArgumentException($"输入长度 {input.Length} 不能被特征维 {fDim} 整除");
        _t = input.Length / fDim;

        // 按 T 分配所有缓冲区：size = T * 存储形状元素积（存储形状时间维=1）
        int n = _g.NBuffers;
        _off = new int[n + 1];
        _count = new int[n];
        int total = 0;
        for (int b = 0; b < n; b++)
        {
            _off[b] = total;
            int c = _t * StoredProduct(b);
            _count[b] = c;
            total += c;
        }
        _off[n] = total;
        _data = new float[total];

        Array.Copy(input, 0, _data, _off[ib], input.Length);

        // 算子级 profiling：MADMOM_PROFILE=1 时按算子类型累计耗时并在 Run 结束打印。
        bool prof = Environment.GetEnvironmentVariable("MADMOM_PROFILE") == "1";
        long[] ticks = prof ? new long[OpNames.Length] : null;
        int[] hits = prof ? new int[OpNames.Length] : null;
        var sw = prof ? Stopwatch.StartNew() : null;

        for (int i = 0; i < _g.NNodes; i++)
        {
            if (prof) sw.Restart();
            Exec(i);
            if (prof) { ticks[_g.Op[i]] += sw.ElapsedTicks; hits[_g.Op[i]]++; }
        }

        if (prof)
        {
            long freq = Stopwatch.Frequency;
            double totalMs = 0;
            for (int op = 0; op < ticks.Length; op++)
            {
                if (hits[op] == 0) continue;
                double ms = ticks[op] * 1000.0 / freq;
                totalMs += ms;
                string name = op < OpNames.Length ? OpNames[op] : $"op{op}";
                Console.Error.WriteLine($"[madmom-profile] {name,-10} hits={hits[op],3} total={ms:F0}ms");
            }
            Console.Error.WriteLine($"[madmom-profile] all-ops total={totalMs:F0}ms  T={_t}");
        }
    }

    public void ReadBuffer(int buffer, float[] dst)
    {
        int c = _count[buffer];
        if (dst.Length < c) throw new ArgumentException("目标数组过小");
        Array.Copy(_data, _off[buffer], dst, 0, c);
    }

    public int BufferLength(int buffer) => _count[buffer];

    // ---------------------------------------------------------------- 调度

    private Span<float> Buf(int b) => _data.AsSpan(_off[b], _count[b]);

    private void Exec(int n)
    {
        switch (_g.Op[n])
        {
            case IrGraph.OpReshape:
            case IrGraph.OpUnsqueeze:
                Buf(_g.In0[n]).CopyTo(Buf(_g.Out[n]));
                break;
            case IrGraph.OpTranspose: Transpose(n); break;
            case IrGraph.OpSlice: Slice(n); break;
            case IrGraph.OpPad: Pad(n); break;
            case IrGraph.OpConcat: Concat(n); break;
            case IrGraph.OpConv: Conv(n); break;
            case IrGraph.OpReduceSum: Reduce(n, 0); break;
            case IrGraph.OpReduceMin: Reduce(n, 1); break;
            case IrGraph.OpReduceMax: Reduce(n, 2); break;
            case IrGraph.OpNeg: Unary(n, 0); break;
            case IrGraph.OpRelu: Unary(n, 1); break;
            case IrGraph.OpSigmoid: Unary(n, 2); break;
            case IrGraph.OpSqrt: Unary(n, 3); break;
            case IrGraph.OpLog: Unary(n, 4); break;
            case IrGraph.OpTanh: Unary(n, 5); break;
            case IrGraph.OpSoftmax: Softmax(n); break;
            case IrGraph.OpMul: Binary(n, 0); break;
            case IrGraph.OpAdd: Binary(n, 1); break;
            case IrGraph.OpSub: Binary(n, 2); break;
            case IrGraph.OpDiv: Binary(n, 3); break;
            case IrGraph.OpEqual: Binary(n, 4); break;
            case IrGraph.OpWhere: Where(n); break;
            case IrGraph.OpLstm: Rnn(n, false, false); break;
            case IrGraph.OpGru: Rnn(n, true, false); break;
            case IrGraph.OpBlstm: Rnn(n, false, true); break;
            case IrGraph.OpDense: Dense(n); break;
            default:
                throw new NotSupportedException("未知算子: " + _g.Op[n]);
        }
    }

    // ------------------------------------------------------------ 形状工具

    private int StoredRank(int b) => _g.BufferRank[b];
    private int StoredDim(int b, int i) => _g.ShapeData[_g.ShapeOff[b] + i];
    private int StoredProduct(int b)
    {
        int r = _g.BufferRank[b], p = 1, off = _g.ShapeOff[b];
        for (int i = 0; i < r; i++) p *= _g.ShapeData[off + i];
        return p;
    }

    /// <summary>运行时形状：把时间维 T 前置到存储形状最前。</summary>
    private int BufferShape(int b, Span<int> dst)
    {
        int r = _g.BufferRank[b];
        dst[0] = _t;
        for (int i = 1; i < r; i++) dst[i] = StoredDim(b, i);
        return r;
    }

    private ReadOnlySpan<float> Operand(int enc)
        => enc >= 0
            ? Buf(enc)
            : _g.WeightData.AsSpan(_g.WeightOff[-enc - 2], _g.WeightCount[-enc - 2]);

    private static void Strides(ReadOnlySpan<int> shape, int rank, Span<int> strides)
    {
        int s = 1;
        for (int i = rank - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
    }

    private static int Reflect(int j, int n)
    {
        if (n <= 1) return 0;
        int period = 2 * (n - 1);
        int m = ((j % period) + period) % period;
        return m >= n ? period - m : m;
    }

    // -------------------------------------------------------------- CNN 算子

    private void Transpose(int n)
    {
        int inB = _g.In0[n], outB = _g.Out[n];
        int p = _g.ListOff[n];
        int r = _g.List[p++];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> perm = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) perm[i] = _g.List[p + i];
        var src = Buf(inB); var dst = Buf(outB);
        int count = _count[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k]; rem /= outShape[k];
                Span<int> istr = stackalloc int[MaxRank];
                Strides(inShape, r, istr);
                si += c * istr[perm[k]];
            }
            dst[o] = src[si];
        }
    }

    private void Slice(int n)
    {
        int inB = _g.In0[n], outB = _g.Out[n];
        int p = _g.ListOff[n];
        int nSpec = _g.List[p++];
        int r = _g.BufferRank[inB];
        Span<int> start = stackalloc int[MaxRank];
        Span<int> step = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) { start[i] = 0; step[i] = 1; }
        for (int s = 0; s < nSpec; s++)
        {
            int ax = _g.List[p++];
            int st = _g.List[p++]; p++;
            int sp = _g.List[p++];
            start[ax] = st; step[ax] = sp;
        }
        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        var src = Buf(inB); var dst = Buf(outB);
        int count = _count[outB];
        Span<int> istr = stackalloc int[MaxRank];
        Strides(inShape, r, istr);
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k]; rem /= outShape[k];
                si += (start[k] + c * step[k]) * istr[k];
            }
            dst[o] = src[si];
        }
    }

    private void Pad(int n)
    {
        int inB = _g.In0[n], outB = _g.Out[n];
        int p = _g.ListOff[n];
        int r = _g.List[p++];
        Span<int> before = stackalloc int[MaxRank];
        Span<int> after = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) before[i] = _g.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) after[i] = _g.List[p + i];
        p += r;
        int mode = _g.List[p];
        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        var src = Buf(inB); var dst = Buf(outB);
        int count = _count[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0; bool inside = true;
            Span<int> istr = stackalloc int[MaxRank];
            Strides(inShape, r, istr);
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k]; rem /= outShape[k];
                int j = c - before[k];
                int dim = inShape[k];
                if (mode == 1) j = Reflect(j, dim);
                else if (mode == 2) j = j < 0 ? 0 : (j >= dim ? dim - 1 : j);
                else if (j < 0 || j >= dim) { inside = false; break; }
                si += j * istr[k];
            }
            dst[o] = inside ? src[si] : 0f;
        }
    }

    private void Concat(int n)
    {
        int outB = _g.Out[n];
        int p = _g.ListOff[n];
        int axis = _g.List[p++];
        int nIns = _g.List[p++];
        if (nIns < 1 || nIns > 8) throw new NotSupportedException($"Concat 输入数不支持: {nIns}");
        int r = _g.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        Span<int> coord = stackalloc int[MaxRank];
        Span<int> segs = stackalloc int[8];
        Span<int> sh = stackalloc int[MaxRank];
        Span<int> st = stackalloc int[MaxRank];
        int total = 0;
        for (int i = 0; i < nIns; i++)
        {
            BufferShape(_g.List[p + 2 + i], sh);
            segs[i] = sh[axis]; total += segs[i];
        }
        if (total != outShape[axis]) throw new InvalidOperationException("Concat 形状不匹配");
        var dst = Buf(outB);
        int count = _count[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o;
            for (int k = r - 1; k >= 0; k--) { coord[k] = rem % outShape[k]; rem /= outShape[k]; }
            int c = coord[axis], acc = 0, chosen = 0;
            for (int i = 0; i < nIns; i++) { if (c < acc + segs[i]) { chosen = i; break; } acc += segs[i]; }
            int ib = _g.List[p + 2 + chosen];
            int rr = BufferShape(ib, sh);
            Strides(sh, rr, st);
            int si = (c - acc) * st[axis];
            for (int k = 0; k < r; k++) if (k != axis) si += coord[k] * st[k];
            dst[o] = Buf(ib)[si];
        }
    }

    private void Conv(int n)
    {
        int inB = _g.In0[n], outB = _g.Out[n];
        int wIdx = -_g.In1[n] - 2;
        int bIdx = _g.In2[n] <= -2 ? -_g.In2[n] - 2 : -1;
        int p = _g.ListOff[n];
        int r = _g.List[p++];
        Span<int> strides = stackalloc int[MaxRank];
        Span<int> padB = stackalloc int[MaxRank];
        Span<int> padA = stackalloc int[MaxRank];
        Span<int> dil = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) strides[i] = _g.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) padB[i] = _g.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) padA[i] = _g.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) dil[i] = _g.List[p + i];
        p += r;
        int group = _g.List[p];
        if (group != 1) throw new NotSupportedException("仅支持 group=1 的卷积");
        Span<int> inShape = stackalloc int[MaxRank];
        Span<int> outShape = stackalloc int[MaxRank];
        Span<int> wShape = stackalloc int[MaxRank];
        int inRank = BufferShape(inB, inShape);
        BufferShape(outB, outShape);
        WeightShape(wIdx, wShape);
        int N = inShape[0], C = inShape[1], M = wShape[0];
        int K = 1; Span<int> ks = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) { ks[i] = wShape[2 + i]; K *= ks[i]; }
        int CK = C * K;
        Span<int> inStrides = stackalloc int[MaxRank]; Strides(inShape, inRank, inStrides);
        Span<int> outStrides = stackalloc int[MaxRank]; Strides(outShape, inRank, outStrides);
        int outSpatial = 1; for (int i = 0; i < r; i++) outSpatial *= outShape[2 + i];
        var x = Buf(inB); var dst = Buf(outB);
        var w = _g.WeightData.AsSpan(_g.WeightOff[wIdx], _g.WeightCount[wIdx]);
        var bias = bIdx >= 0 ? _g.WeightData.AsSpan(_g.WeightOff[bIdx], _g.WeightCount[bIdx]) : ReadOnlySpan<float>.Empty;
        Span<float> acc = stackalloc float[M];
        Span<int> oc = stackalloc int[MaxRank];
        for (int nb = 0; nb < N; nb++)
        {
            int inNBase = nb * inStrides[0], outNBase = nb * outStrides[0];
            for (int o = 0; o < outSpatial; o++)
            {
                int rem = o;
                for (int i = r - 1; i >= 0; i--) { oc[i] = rem % outShape[2 + i]; rem /= outShape[2 + i]; }
                for (int m = 0; m < M; m++) acc[m] = bIdx >= 0 ? bias[m] : 0f;
                for (int c = 0; c < C; c++)
                {
                    int inCBase = inNBase + c * inStrides[1];
                    int wCBase = c * K;
                    for (int kk = 0; kk < K; kk++)
                    {
                        int tt = kk, inOff = inCBase; bool inside = true;
                        for (int i = r - 1; i >= 0; i--)
                        {
                            int kc = tt % ks[i]; tt /= ks[i];
                            int src = oc[i] * strides[i] - padB[i] + kc * dil[i];
                            if (src < 0 || src >= inShape[2 + i]) { inside = false; break; }
                            inOff += src * inStrides[2 + i];
                        }
                        if (!inside) continue;
                        float xv = x[inOff];
                        int wBase = wCBase + kk;
                        for (int m = 0; m < M; m++) acc[m] += xv * w[m * CK + wBase];
                    }
                }
                int outBase = outNBase + o;
                for (int m = 0; m < M; m++) dst[outBase + m * outStrides[1]] = acc[m];
            }
        }
    }

    private void Reduce(int n, int kind)
    {
        int inB = _g.In0[n], outB = _g.Out[n];
        int p = _g.ListOff[n]; p++;
        int nAx = _g.List[p++];
        int inRank = _g.BufferRank[inB];
        Span<bool> reduced = stackalloc bool[MaxRank];
        for (int i = 0; i < inRank; i++) reduced[i] = false;
        for (int i = 0; i < nAx; i++) reduced[_g.List[p + i]] = true;
        Span<int> inShape = stackalloc int[MaxRank];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        BufferShape(outB, outShape);
        int outRank = _g.BufferRank[outB];
        Span<int> outStrides = stackalloc int[MaxRank]; Strides(outShape, outRank, outStrides);
        Span<int> map = stackalloc int[MaxRank];
        int oa = 0;
        for (int k = 0; k < inRank; k++) map[k] = reduced[k] ? -1 : oa++;
        var src = Buf(inB); var dst = Buf(outB);
        int outCount = _count[outB];
        float init = kind == 0 ? 0f : (kind == 1 ? float.PositiveInfinity : float.NegativeInfinity);
        for (int i = 0; i < outCount; i++) dst[i] = init;
        int count = _count[inB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, oi = 0;
            for (int k = inRank - 1; k >= 0; k--)
            {
                int c = rem % inShape[k]; rem /= inShape[k];
                if (map[k] >= 0) oi += c * outStrides[map[k]];
            }
            float v = src[o];
            if (kind == 0) dst[oi] += v;
            else if (kind == 1) { if (v < dst[oi]) dst[oi] = v; }
            else { if (v > dst[oi]) dst[oi] = v; }
        }
    }

    private void Unary(int n, int kind)
    {
        var src = Buf(_g.In0[n]); var dst = Buf(_g.Out[n]);
        switch (kind)
        {
            case 0: for (int i = 0; i < src.Length; i++) dst[i] = -src[i]; break;
            case 1: for (int i = 0; i < src.Length; i++) dst[i] = src[i] > 0f ? src[i] : 0f; break;
            case 2: for (int i = 0; i < src.Length; i++) dst[i] = 1f / (1f + MathF.Exp(-src[i])); break;
            case 3: for (int i = 0; i < src.Length; i++) dst[i] = MathF.Sqrt(src[i]); break;
            case 5: for (int i = 0; i < src.Length; i++) dst[i] = MathF.Tanh(src[i]); break;
            default: for (int i = 0; i < src.Length; i++) dst[i] = MathF.Log(src[i]); break;
        }
    }

    private void Binary(int n, int kind)
    {
        int outB = _g.Out[n]; int outRank = _g.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank]; BufferShape(outB, outShape);
        Span<int> eA = stackalloc int[MaxRank]; Span<int> eB = stackalloc int[MaxRank];
        EffStrides(_g.In0[n], outRank, eA); EffStrides(_g.In1[n], outRank, eB);
        var a = Operand(_g.In0[n]); var b = Operand(_g.In1[n]); var dst = Buf(outB);
        int count = _count[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, ia = 0, ib = 0;
            for (int k = outRank - 1; k >= 0; k--) { int c = rem % outShape[k]; rem /= outShape[k]; ia += c * eA[k]; ib += c * eB[k]; }
            float x = a[ia], y = b[ib];
            dst[o] = kind switch { 0 => x * y, 1 => x + y, 2 => x - y, 3 => x / y, _ => x == y ? 1f : 0f };
        }
    }

    private void Where(int n)
    {
        int outB = _g.Out[n]; int outRank = _g.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank]; BufferShape(outB, outShape);
        Span<int> eC = stackalloc int[MaxRank]; Span<int> eX = stackalloc int[MaxRank]; Span<int> eY = stackalloc int[MaxRank];
        EffStrides(_g.In0[n], outRank, eC); EffStrides(_g.In1[n], outRank, eX); EffStrides(_g.In2[n], outRank, eY);
        var cond = Operand(_g.In0[n]); var xa = Operand(_g.In1[n]); var ya = Operand(_g.In2[n]); var dst = Buf(outB);
        int count = _count[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, ic = 0, ix = 0, iy = 0;
            for (int k = outRank - 1; k >= 0; k--) { int c = rem % outShape[k]; rem /= outShape[k]; ic += c * eC[k]; ix += c * eX[k]; iy += c * eY[k]; }
            dst[o] = cond[ic] != 0f ? xa[ix] : ya[iy];
        }
    }

    private void WeightShape(int w, Span<int> dst)
    {
        int r = _g.WeightRank[w]; int off = _g.WeightShapeOff[w];
        for (int i = 0; i < r; i++) dst[i] = _g.WeightShapeData[off + i];
    }

    private void EffStrides(int enc, int outRank, Span<int> eff)
    {
        Span<int> shape = stackalloc int[MaxRank];
        int r = enc >= 0 ? BufferShape(enc, shape) : WeightRankShape(enc, shape);
        int shift = outRank - r; int s = 1;
        for (int k = r - 1; k >= 0; k--) { eff[shift + k] = shape[k] == 1 ? 0 : s; s *= shape[k]; }
        for (int k = 0; k < shift; k++) eff[k] = 0;
    }

    private int WeightRankShape(int enc, Span<int> dst)
    {
        int w = -enc - 2;
        int r = _g.WeightRank[w];
        int off = _g.WeightShapeOff[w];
        for (int i = 0; i < r; i++) dst[i] = _g.WeightShapeData[off + i];
        return r;
    }

    // -------------------------------------------------------------- RNN 算子

    /// <summary>
    /// 执行 LSTM / GRU / BLSTM。
    /// 约定（与本仓库 export_madmom_ir.py 必须一致）：
    ///  - 输入 x 形状 (T, F)；输出 (T, H) 单向 / (T, 2H) 双向。
    ///  - 权重按名字取（node 为该 RNN 在图里的节点序号，支持多层堆叠）：
    ///    单向："{pfx}{node}x"=Wx(4HxF 或 3HxF)、"{pfx}{node}h"=Wh、"{pfx}{node}b"=b；
    ///    双向：再加 f/b 方向后缀，即 Wxf/Whf/bf、Wxb/Whb/bb（如 "lstm_3fx"）。
    ///  - LSTM 门序 [i,f,c,o]（madmom / Lasagne 默认）；GRU 门序 [z,r,n]。
    /// </summary>
    private void Rnn(int n, bool gru, bool bidirectional)
    {
        int xB = _g.In0[n];
        Span<int> xsh = stackalloc int[MaxRank];
        int xr = BufferShape(xB, xsh);
        int T = xsh[0];
        int F = xsh[xr - 1];
        int p = _g.ListOff[n];
        int H = _g.List[p++];
        int bidir = _g.List[p++];
        int retSeq = _g.List[p++];
        if (bidir != (bidirectional ? 1 : 0))
            throw new InvalidOperationException("RNN 双向标志不匹配");

        int xBase = _off[xB];   // 取底层数组，便于点积走 SIMD（netstandard2.1 的 Vector 只吃 Span）
        int outB = _g.Out[n];
        Span<int> osh = stackalloc int[MaxRank];
        int orank = BufferShape(outB, osh);
        var dst = Buf(outB);
        int outF = osh[orank - 1];

        if (!bidirectional)
        {
            var (Wx, Wh, b, pe) = LoadRnnWeights("", n, gru);
            var h = new float[H]; var c = new float[H];
            LoadState(n, h, c);
            Span<float> outRow = stackalloc float[H];
            for (int t = 0; t < T; t++)
            {
                StepRnn(gru, Wx, Wh, b, pe, _data, xBase + t * F, F, h, c, outRow);
                if (retSeq != 0)
                    for (int j = 0; j < H; j++) dst[t * outF + j] = outRow[j];
            }
            if (retSeq == 0)
                for (int j = 0; j < H; j++) dst[j] = h[j];
            SaveState(n, h, c);
        }
        else
        {
            // Unity：Burst 整层内核（双向两线程并行）。仅覆盖完整序列输出（retSeq!=0）。
            if (Seams.Blstm != null && retSeq != 0)
            {
                var (Wxf0, Whf0, bf0, pef0) = LoadRnnWeights("f", n, gru);
                var (Wxb0, Whb0, bb0, peb0) = LoadRnnWeights("b", n, gru);
                var hf0 = new float[H]; var cf0 = new float[H];
                LoadState(n, hf0, cf0);
                Seams.Blstm.Run(_data, xBase, T, F, Wxf0, Whf0, bf0, Wxb0, Whb0, bb0,
                    pef0, H, dst, outF, hf0, cf0);
                SaveState(n, hf0, cf0);
                return;
            }

            var (Wxf, Whf, bf, pef) = LoadRnnWeights("f", n, gru);
            var (Wxb, Whb, bb, peb) = LoadRnnWeights("b", n, gru);
            var hf = new float[H]; var cf = new float[H];
            LoadState(n, hf, cf);     // 前向：承接上一块末尾状态
            // 后向：每块从块末尾重新开始（无法在线），块长足够时与全量差异可忽略
            var hb = new float[H]; var cb = new float[H];
            Span<float> rf = stackalloc float[H];
            Span<float> rb = stackalloc float[H];
            // 前向：位置 t 喂入 x[t]，hf 累积 x[0..t] → 与 madmom fwd_layer 一致。
            // 后向：沿反向累积（喂入 x[T-1], x[T-2], …），但把结果写回 srcIdx=t 处，
            // 等价于 madmom 的 bwd_layer(data[::-1]) 后做 bwd[::-1] 反转（位置对齐）。
            for (int t = 0; t < T; t++)
            {
                int srcIdx = T - 1 - t;
                StepRnn(gru, Wxf, Whf, bf, pef, _data, xBase + t * F, F, hf, cf, rf);
                for (int j = 0; j < H; j++) dst[t * outF + j] = rf[j];
                StepRnn(gru, Wxb, Whb, bb, peb, _data, xBase + srcIdx * F, F, hb, cb, rb);
                for (int j = 0; j < H; j++) dst[srcIdx * outF + H + j] = rb[j];
            }
            SaveState(n, hf, cf);
        }
    }

    private (float[] Wx, float[] Wh, float[] b, float[] pe) LoadRnnWeights(string dir, int node, bool gru)
    {
        string pfx = gru ? "gru_" : "lstm_";
        var pe = gru ? Array.Empty<float>() : _g.W(pfx + node + dir + "p");
        return (_g.W(pfx + node + dir + "x"), _g.W(pfx + node + dir + "h"),
                _g.W(pfx + node + dir + "b"), pe);
    }

    private static void StepRnn(bool gru, float[] Wx, float[] Wh, float[] b, float[] pe,
        float[] x, int xOff, int F, float[] h, float[] c, Span<float> outRow)
    {
        int gates = gru ? 3 : 4;
        int H = h.Length;
        bool hasPe = pe.Length > 0;
        Span<float> pre = stackalloc float[gates * H];
        // 门控预激活 = Wx·x + Wh·h + b。两个点积占 RNN 前向绝大部分算力，走 SIMD。
        for (int k = 0; k < gates * H; k++)
        {
            float s = b[k];
            s += Simd.Dot(Wx, k * F, x, xOff, F);
            s += Simd.Dot(Wh, k * H, h, 0, H);
            pre[k] = s;
        }
        if (gru)
        {
            Span<float> z = stackalloc float[H];
            Span<float> r = stackalloc float[H];
            Span<float> nn = stackalloc float[H];
            for (int j = 0; j < H; j++) { z[j] = Sigmoid(pre[j]); r[j] = Sigmoid(pre[H + j]); }
            for (int j = 0; j < H; j++)
            {
                float nPre = pre[2 * H + j];
                for (int k2 = 0; k2 < H; k2++) nPre += r[j] * Wh[(2 * H + j) * H + k2] * h[k2];
                nn[j] = MathF.Tanh(nPre);
            }
            for (int j = 0; j < H; j++) outRow[j] = (1 - z[j]) * nn[j] + z[j] * h[j];
            for (int j = 0; j < H; j++) h[j] = outRow[j];
        }
        else
        {
            Span<float> ci = stackalloc float[H];
            Span<float> co = stackalloc float[H];
            for (int j = 0; j < H; j++)
            {
                ci[j] = Sigmoid(pre[j] + (hasPe ? pe[j] * c[j] : 0f));
                co[j] = Sigmoid(pre[H + j] + (hasPe ? pe[H + j] * c[j] : 0f));
            }
            Span<float> cIn = stackalloc float[H];
            for (int j = 0; j < H; j++) cIn[j] = MathF.Tanh(pre[2 * H + j]);
            for (int j = 0; j < H; j++) c[j] = co[j] * c[j] + ci[j] * cIn[j];
            Span<float> o = stackalloc float[H];
            for (int j = 0; j < H; j++)
                o[j] = Sigmoid(pre[3 * H + j] + (hasPe ? pe[3 * H + j] * c[j] : 0f));
            for (int j = 0; j < H; j++)
            {
                outRow[j] = o[j] * MathF.Tanh(c[j]);
                h[j] = outRow[j];
            }
        }
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    // -------------------------------------------------------------- Dense 算子

    /// <summary>
    /// 全连接：y = x·W + b。权重按名字取 dense{node}_x( out×in )、dense{node}_b( out )。
    /// 输入 x 形状 (T,F)；输出 (T, out)。支持同一图内多个 Dense 节点。
    /// </summary>
    private void Dense(int n)
    {
        int xB = _g.In0[n];
        Span<int> xsh = stackalloc int[MaxRank];
        int xr = BufferShape(xB, xsh);
        int T = xsh[0];
        int F = xsh[xr - 1];
        var W = _g.W("dense" + n + "_x");
        var b = _g.W("dense" + n + "_b");
        int outDim = b.Length;
        var x = Buf(xB);
        var dst = Buf(_g.Out[n]);
        Span<int> osh = stackalloc int[MaxRank];
        int orank = BufferShape(_g.Out[n], osh);
        int outF = osh[orank - 1];
        for (int t = 0; t < T; t++)
            for (int o = 0; o < outDim; o++)
            {
                float s = b[o];
                int baseK = o * F;
                int xo = t * F;
                for (int j = 0; j < F; j++) s += x[xo + j] * W[baseK + j];
                dst[t * outF + o] = s;
            }
    }

    /// <summary>
    /// 稳定 softmax：沿缓冲区最后一维（类别维）逐行归一化，与 madmom
    /// <c>activations.softmax</c>（axis=1，先减行最大值）一致。
    /// </summary>
    private void Softmax(int n)
    {
        int inB = _g.In0[n];
        Span<int> xsh = stackalloc int[MaxRank];
        int xr = BufferShape(inB, xsh);
        int C = xsh[xr - 1];
        int total = _count[inB];
        int outer = total / C;
        var x = Buf(inB);
        var dst = Buf(_g.Out[n]);
        for (int o = 0; o < outer; o++)
        {
            int baseIdx = o * C;
            float mx = x[baseIdx];
            for (int c = 1; c < C; c++) if (x[baseIdx + c] > mx) mx = x[baseIdx + c];
            float sum = 0f;
            for (int c = 0; c < C; c++)
            {
                float e = MathF.Exp(x[baseIdx + c] - mx);
                dst[baseIdx + c] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int c = 0; c < C; c++) dst[baseIdx + c] *= inv;
        }
    }
}
