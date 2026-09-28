using System;

namespace BasicPitch;

/// <summary>
/// Basic Pitch 前向引擎的抽象。
///
/// 桌面端默认使用内置的纯 C# 引擎 <see cref="NmpEngine"/>；
/// Unity / 移动端可传入 Burst 加速实现（见 UnityExample 的 NmpBurstEngine），
/// 两者数值等价，替换后 <see cref="BasicPitchConverter"/> 的其余流程完全不变。
/// </summary>
public interface INmpForwardEngine
{
    /// <summary>模型输入缓冲区所需的样本数（43844）。</summary>
    int InputLength { get; }

    /// <summary>执行一次前向计算，<paramref name="input"/> 长度须等于 <see cref="InputLength"/>。</summary>
    void Run(float[] input);

    /// <summary>返回指定输出缓冲区的元素个数。</summary>
    int BufferLength(int buffer);

    /// <summary>把指定输出缓冲区的数据复制到 <paramref name="dst"/>。</summary>
    void ReadBuffer(int buffer, float[] dst);
}

/// <summary>
/// 纯 C# 实现的 Basic Pitch (nmp) 前向推理引擎。
///
/// 图拓扑与权重常量来自 <c>NmpIr.g.cs</c>（由 nmp.onnx 常量折叠后生成），
/// 因此不依赖 ONNX Runtime、System.Numerics.Tensors 或任何第三方库，
/// 可在 Windows / iOS / Android（含 Unity Burst 环境）下运行。
///
/// 缓冲区策略：243 个中间张量一次性分配在一块连续 float 数组中并跨窗口复用，
/// 避免每帧分配带来的 GC 抖动（总计约 33 MB）。
/// </summary>
internal sealed class NmpEngine : INmpForwardEngine
{
    private const int MaxRank = 5;

    private readonly float[] _data;   // 所有缓冲区共享的大数组
    private readonly int[] _off;      // 每个缓冲区在 _data 中的起始偏移

    public NmpEngine()
    {
        _off = new int[NmpIr.NBuffers + 1];
        int total = 0;
        for (int i = 0; i < NmpIr.NBuffers; i++)
        {
            _off[i] = total;
            total += NmpIr.BufferCount[i];
        }
        _off[NmpIr.NBuffers] = total;
        _data = new float[total];
    }

    /// <summary>模型输入缓冲区所需的样本数（43844）。</summary>
    public int InputLength => NmpIr.BufferCount[NmpIr.InputBuffer];

    /// <summary>执行一次前向计算。</summary>
    public void Run(float[] input)
    {
        int ib = NmpIr.InputBuffer;
        int n = NmpIr.BufferCount[ib];
        if (input.Length != n)
            throw new ArgumentException($"输入长度应为 {n}，实际为 {input.Length}");

        Array.Copy(input, 0, _data, _off[ib], n);
        for (int i = 0; i < NmpIr.NNodes; i++)
            Exec(i);
    }

    /// <summary>把指定缓冲区的数据复制到 <paramref name="dst"/>。</summary>
    public void ReadBuffer(int buffer, float[] dst)
        => Array.Copy(_data, _off[buffer], dst, 0, NmpIr.BufferCount[buffer]);

    public int BufferLength(int buffer) => NmpIr.BufferCount[buffer];

    // ------------------------------------------------------------------ 调度

    private Span<float> Buf(int b) => _data.AsSpan(_off[b], NmpIr.BufferCount[b]);

    private static int Attr(int node, int k) => NmpIr.List[NmpIr.ListOff[node] + k];

    private void Exec(int n)
    {
        switch (NmpIr.Op[n])
        {
            case NmpIr.Reshape:
            case NmpIr.Unsqueeze:
                Buf(NmpIr.In0[n]).CopyTo(Buf(NmpIr.Out[n]));
                break;
            case NmpIr.Transpose: Transpose(n); break;
            case NmpIr.Slice: Slice(n); break;
            case NmpIr.Pad: Pad(n); break;
            case NmpIr.Concat: Concat(n); break;
            case NmpIr.Conv: Conv(n); break;
            case NmpIr.ReduceSum: Reduce(n, 0); break;
            case NmpIr.ReduceMin: Reduce(n, 1); break;
            case NmpIr.ReduceMax: Reduce(n, 2); break;
            case NmpIr.Neg: Unary(n, 0); break;
            case NmpIr.Relu: Unary(n, 1); break;
            case NmpIr.Sigmoid: Unary(n, 2); break;
            case NmpIr.Sqrt: Unary(n, 3); break;
            case NmpIr.Log: Unary(n, 4); break;
            case NmpIr.Mul: Binary(n, 0); break;
            case NmpIr.Add: Binary(n, 1); break;
            case NmpIr.Sub: Binary(n, 2); break;
            case NmpIr.Div: Binary(n, 3); break;
            case NmpIr.Equal: Binary(n, 4); break;
            case NmpIr.Where: Where(n); break;
            default:
                throw new NotSupportedException("未知算子: " + NmpIr.Op[n]);
        }
    }

    // -------------------------------------------------------------- 形状工具

    private static int BufferShape(int b, Span<int> dst)
    {
        int r = NmpIr.BufferRank[b];
        int off = NmpIr.ShapeOff[b];
        for (int i = 0; i < r; i++) dst[i] = NmpIr.ShapeData[off + i];
        return r;
    }

    private static int WeightShape(int w, Span<int> dst)
    {
        int r = NmpIr.WeightRank[w];
        int off = NmpIr.WeightShapeOff[w];
        for (int i = 0; i < r; i++) dst[i] = NmpIr.WeightShapeData[off + i];
        return r;
    }

    /// <summary>取任一输入（缓冲区或权重常量）的形状，返回秩。</summary>
    private static int ShapeOf(int enc, Span<int> dst)
        => enc >= 0 ? BufferShape(enc, dst) : WeightShape(-enc - 2, dst);

    private ReadOnlySpan<float> Operand(int enc)
        => enc >= 0
            ? Buf(enc)
            : NmpIr.WeightData.AsSpan(NmpIr.WeightOff[-enc - 2], NmpIr.WeightCount[-enc - 2]);

    private static void Strides(ReadOnlySpan<int> shape, int rank, Span<int> strides)
    {
        int s = 1;
        for (int i = rank - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
    }

    /// <summary>把任意形状的输入轴在 outRank 维输出上的有效步长（广播时该轴为 0）。</summary>
    private static void EffStrides(int enc, int outRank, Span<int> eff)
    {
        Span<int> shape = stackalloc int[MaxRank];
        int r = ShapeOf(enc, shape);
        int shift = outRank - r;
        int s = 1;
        for (int k = r - 1; k >= 0; k--)
        {
            eff[shift + k] = shape[k] == 1 ? 0 : s;
            s *= shape[k];
        }
        for (int k = 0; k < shift; k++) eff[k] = 0;
    }

    /// <summary>numpy/ONNX 的 reflect 镜像索引（不重复边界值）。</summary>
    private static int Reflect(int j, int n)
    {
        if (n <= 1) return 0;
        int period = 2 * (n - 1);
        int m = ((j % period) + period) % period;
        return m >= n ? period - m : m;
    }

    // ------------------------------------------------------------------ 算子

    private void Transpose(int n)
    {
        int inB = NmpIr.In0[n], outB = NmpIr.Out[n];
        int p = NmpIr.ListOff[n];
        int r = NmpIr.List[p++];

        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        Span<int> outStrides = stackalloc int[MaxRank];
        Strides(outShape, r, outStrides);

        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> inStrides = stackalloc int[MaxRank];
        Strides(inShape, r, inStrides);

        Span<int> perm = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) perm[i] = NmpIr.List[p + i];

        var src = Buf(inB);
        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k];
                rem /= outShape[k];
                si += c * inStrides[perm[k]];
            }
            dst[o] = src[si];
        }
    }

    private void Slice(int n)
    {
        int inB = NmpIr.In0[n], outB = NmpIr.Out[n];
        int p = NmpIr.ListOff[n];
        int nSpec = NmpIr.List[p++];
        int r = NmpIr.BufferRank[inB];

        Span<int> start = stackalloc int[MaxRank];
        Span<int> step = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) { start[i] = 0; step[i] = 1; }
        for (int s = 0; s < nSpec; s++)
        {
            int ax = NmpIr.List[p++];
            int st = NmpIr.List[p++];
            p++;                        // end 已归一化，复制时无需使用
            int sp = NmpIr.List[p++];
            start[ax] = st;
            step[ax] = sp;
        }

        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> inStrides = stackalloc int[MaxRank];
        Strides(inShape, r, inStrides);

        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);

        var src = Buf(inB);
        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k];
                rem /= outShape[k];
                si += (start[k] + c * step[k]) * inStrides[k];
            }
            dst[o] = src[si];
        }
    }

    private void Pad(int n)
    {
        int inB = NmpIr.In0[n], outB = NmpIr.Out[n];
        int p = NmpIr.ListOff[n];
        int r = NmpIr.List[p++];

        Span<int> before = stackalloc int[MaxRank];
        Span<int> after = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) before[i] = NmpIr.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) after[i] = NmpIr.List[p + i];
        p += r;
        int mode = NmpIr.List[p];    // 0 constant / 1 reflect / 2 edge

        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> inStrides = stackalloc int[MaxRank];
        Strides(inShape, r, inStrides);

        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);

        var src = Buf(inB);
        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, si = 0;
            bool inside = true;
            for (int k = r - 1; k >= 0; k--)
            {
                int c = rem % outShape[k];
                rem /= outShape[k];
                int j = c - before[k];
                int dim = inShape[k];
                if (mode == 1) j = Reflect(j, dim);
                else if (mode == 2) j = j < 0 ? 0 : (j >= dim ? dim - 1 : j);
                else if (j < 0 || j >= dim) { inside = false; break; }
                si += j * inStrides[k];
            }
            dst[o] = inside ? src[si] : 0f;
        }
    }

    private void Concat(int n)
    {
        int outB = NmpIr.Out[n];
        int p = NmpIr.ListOff[n];
        int axis = NmpIr.List[p];
        int nIns = NmpIr.List[p + 1];        // 紧随其后是各输入缓冲区索引
        if (nIns < 1 || nIns > 8) throw new NotSupportedException($"Concat 输入数不支持: {nIns}");

        int r = NmpIr.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        Span<int> coord = stackalloc int[MaxRank];
        Span<int> segs = stackalloc int[8];      // 各输入在拼接轴上的长度
        Span<int> sh = stackalloc int[MaxRank];
        Span<int> st = stackalloc int[MaxRank];

        int total = 0;
        for (int i = 0; i < nIns; i++)
        {
            BufferShape(NmpIr.List[p + 2 + i], sh);
            segs[i] = sh[axis];
            total += segs[i];
        }
        if (total != outShape[axis])
            throw new InvalidOperationException("Concat 形状不匹配");

        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];
        for (int o = 0; o < count; o++)
        {
            int rem = o;
            for (int k = r - 1; k >= 0; k--) { coord[k] = rem % outShape[k]; rem /= outShape[k]; }

            int c = coord[axis];
            int acc = 0, chosen = 0;
            for (int i = 0; i < nIns; i++)
            {
                if (c < acc + segs[i]) { chosen = i; break; }
                acc += segs[i];
            }

            int ib = NmpIr.List[p + 2 + chosen];
            int rr = BufferShape(ib, sh);
            Strides(sh, rr, st);
            int si = (c - acc) * st[axis];
            for (int k = 0; k < r; k++)
                if (k != axis) si += coord[k] * st[k];
            dst[o] = Buf(ib)[si];
        }
    }

    private void Conv(int n)
    {
        int inB = NmpIr.In0[n], outB = NmpIr.Out[n];
        int wIdx = -NmpIr.In1[n] - 2;
        int bIdx = NmpIr.In2[n] <= -2 ? -NmpIr.In2[n] - 2 : -1;

        int p = NmpIr.ListOff[n];
        int r = NmpIr.List[p++];
        Span<int> strides = stackalloc int[MaxRank];
        Span<int> padB = stackalloc int[MaxRank];
        Span<int> padA = stackalloc int[MaxRank];
        Span<int> dil = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) strides[i] = NmpIr.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) padB[i] = NmpIr.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) padA[i] = NmpIr.List[p + i];
        p += r;
        for (int i = 0; i < r; i++) dil[i] = NmpIr.List[p + i];
        p += r;
        int group = NmpIr.List[p];
        if (group != 1) throw new NotSupportedException("仅支持 group=1 的卷积");

        Span<int> inShape = stackalloc int[MaxRank];
        Span<int> outShape = stackalloc int[MaxRank];
        Span<int> wShape = stackalloc int[MaxRank];
        int inRank = BufferShape(inB, inShape);
        BufferShape(outB, outShape);
        WeightShape(wIdx, wShape);

        int N = inShape[0], C = inShape[1];
        int M = wShape[0];
        int K = 1;
        Span<int> ks = stackalloc int[MaxRank];
        for (int i = 0; i < r; i++) { ks[i] = wShape[2 + i]; K *= ks[i]; }
        int CK = C * K;

        Span<int> inStrides = stackalloc int[MaxRank];
        Strides(inShape, inRank, inStrides);
        Span<int> outStrides = stackalloc int[MaxRank];
        Strides(outShape, inRank, outStrides);

        int outSpatial = 1;
        for (int i = 0; i < r; i++) outSpatial *= outShape[2 + i];

        var x = Buf(inB);
        var dst = Buf(outB);
        var w = NmpIr.WeightData.AsSpan(NmpIr.WeightOff[wIdx], NmpIr.WeightCount[wIdx]);
        var bias = bIdx >= 0
            ? NmpIr.WeightData.AsSpan(NmpIr.WeightOff[bIdx], NmpIr.WeightCount[bIdx])
            : ReadOnlySpan<float>.Empty;

        Span<float> acc = stackalloc float[M];
        Span<int> oc = stackalloc int[MaxRank];

        for (int nb = 0; nb < N; nb++)
        {
            int inNBase = nb * inStrides[0];
            int outNBase = nb * outStrides[0];
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
                        int tt = kk, inOff = inCBase;
                        bool inside = true;
                        for (int i = r - 1; i >= 0; i--)
                        {
                            int kc = tt % ks[i];
                            tt /= ks[i];
                            int src = oc[i] * strides[i] - padB[i] + kc * dil[i];
                            if (src < 0 || src >= inShape[2 + i]) { inside = false; break; }
                            inOff += src * inStrides[2 + i];
                        }
                        if (!inside) continue;

                        float xv = x[inOff];
                        int wBase = wCBase + kk;
                        for (int m = 0; m < M; m++)
                            acc[m] += xv * w[m * CK + wBase];
                    }
                }

                int outBase = outNBase + o;
                for (int m = 0; m < M; m++)
                    dst[outBase + m * outStrides[1]] = acc[m];
            }
        }
    }

    private void Reduce(int n, int kind)   // 0=sum 1=min 2=max
    {
        int inB = NmpIr.In0[n], outB = NmpIr.Out[n];
        int p = NmpIr.ListOff[n];
        p++;                                  // keepdims，两种取值下边的索引公式都成立
        int nAx = NmpIr.List[p++];

        int inRank = NmpIr.BufferRank[inB];
        Span<bool> reduced = stackalloc bool[MaxRank];
        for (int i = 0; i < inRank; i++) reduced[i] = false;
        for (int i = 0; i < nAx; i++) reduced[NmpIr.List[p + i]] = true;

        Span<int> inShape = stackalloc int[MaxRank];
        BufferShape(inB, inShape);
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);
        int outRank = NmpIr.BufferRank[outB];
        Span<int> outStrides = stackalloc int[MaxRank];
        Strides(outShape, outRank, outStrides);

        Span<int> map = stackalloc int[MaxRank];
        int oa = 0;
        for (int k = 0; k < inRank; k++) map[k] = reduced[k] ? -1 : oa++;

        var src = Buf(inB);
        var dst = Buf(outB);
        int outCount = NmpIr.BufferCount[outB];
        float init = kind == 0 ? 0f : (kind == 1 ? float.PositiveInfinity : float.NegativeInfinity);
        for (int i = 0; i < outCount; i++) dst[i] = init;

        int count = NmpIr.BufferCount[inB];
        for (int o = 0; o < count; o++)
        {
            int rem = o, oi = 0;
            for (int k = inRank - 1; k >= 0; k--)
            {
                int c = rem % inShape[k];
                rem /= inShape[k];
                if (map[k] >= 0) oi += c * outStrides[map[k]];
            }
            float v = src[o];
            if (kind == 0) dst[oi] += v;
            else if (kind == 1) { if (v < dst[oi]) dst[oi] = v; }
            else { if (v > dst[oi]) dst[oi] = v; }
        }
    }

    private void Unary(int n, int kind)   // 0=neg 1=relu 2=sigmoid 3=sqrt 4=log
    {
        var src = Buf(NmpIr.In0[n]);
        var dst = Buf(NmpIr.Out[n]);
        switch (kind)
        {
            case 0: for (int i = 0; i < src.Length; i++) dst[i] = -src[i]; break;
            case 1: for (int i = 0; i < src.Length; i++) dst[i] = src[i] > 0f ? src[i] : 0f; break;
            case 2: for (int i = 0; i < src.Length; i++) dst[i] = 1f / (1f + MathF.Exp(-src[i])); break;
            case 3: for (int i = 0; i < src.Length; i++) dst[i] = MathF.Sqrt(src[i]); break;
            default: for (int i = 0; i < src.Length; i++) dst[i] = MathF.Log(src[i]); break;
        }
    }

    private void Binary(int n, int kind)  // 0=mul 1=add 2=sub 3=div 4=equal
    {
        int outB = NmpIr.Out[n];
        int outRank = NmpIr.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);

        Span<int> eA = stackalloc int[MaxRank];
        Span<int> eB = stackalloc int[MaxRank];
        EffStrides(NmpIr.In0[n], outRank, eA);
        EffStrides(NmpIr.In1[n], outRank, eB);

        var a = Operand(NmpIr.In0[n]);
        var b = Operand(NmpIr.In1[n]);
        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];

        for (int o = 0; o < count; o++)
        {
            int rem = o, ia = 0, ib = 0;
            for (int k = outRank - 1; k >= 0; k--)
            {
                int c = rem % outShape[k];
                rem /= outShape[k];
                ia += c * eA[k];
                ib += c * eB[k];
            }
            float x = a[ia], y = b[ib];
            switch (kind)
            {
                case 0: dst[o] = x * y; break;
                case 1: dst[o] = x + y; break;
                case 2: dst[o] = x - y; break;
                case 3: dst[o] = x / y; break;
                default: dst[o] = x == y ? 1f : 0f; break;
            }
        }
    }

    private void Where(int n)
    {
        int outB = NmpIr.Out[n];
        int outRank = NmpIr.BufferRank[outB];
        Span<int> outShape = stackalloc int[MaxRank];
        BufferShape(outB, outShape);

        Span<int> eC = stackalloc int[MaxRank];
        Span<int> eX = stackalloc int[MaxRank];
        Span<int> eY = stackalloc int[MaxRank];
        EffStrides(NmpIr.In0[n], outRank, eC);
        EffStrides(NmpIr.In1[n], outRank, eX);
        EffStrides(NmpIr.In2[n], outRank, eY);

        var cond = Operand(NmpIr.In0[n]);
        var xa = Operand(NmpIr.In1[n]);
        var ya = Operand(NmpIr.In2[n]);
        var dst = Buf(outB);
        int count = NmpIr.BufferCount[outB];

        for (int o = 0; o < count; o++)
        {
            int rem = o, ic = 0, ix = 0, iy = 0;
            for (int k = outRank - 1; k >= 0; k--)
            {
                int c = rem % outShape[k];
                rem /= outShape[k];
                ic += c * eC[k];
                ix += c * eX[k];
                iy += c * eY[k];
            }
            dst[o] = cond[ic] != 0f ? xa[ix] : ya[iy];
        }
    }
}