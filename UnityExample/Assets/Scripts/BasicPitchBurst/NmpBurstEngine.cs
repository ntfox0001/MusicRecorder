using System;
using BasicPitch;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace BasicPitch.Burst
{
    /// <summary>
    /// 基于 Unity Burst 的 Basic Pitch (nmp) 前向引擎，面向移动端（iOS / Android）。
    ///
    /// 与桌面端的纯 C# 引擎 <c>NmpEngine</c> 数值等价，区别只在执行方式：
    ///   - 243 个中间张量存放在一块连续 NativeArray（约 32.85 MB），一次性分配、跨窗口复用；
    ///   - 逐元素 / 布局 / 规约算子经 [BurstCompile] 编译为 SIMD 机器码，并按输出元素并行；
    ///   - 卷积（约占总计算量 70%）按输出元素多维并行，走 Job System 吃满所有核心。
    ///
    /// 使用前提：
    ///   1. Package Manager 安装 Burst（com.unity.burst）；
    ///   2. BasicPitch.dll 放入 Assets/Plugins/（本引擎复用其公开的 NmpIr 图数据与后处理）。
    ///
    /// 用法：
    /// <code>
    /// using var converter = new BasicPitchConverter(new NmpBurstEngine());
    /// var notes = converter.Convert(samples, sampleRate);
    /// </code>
    /// </summary>
    public sealed class NmpBurstEngine : INmpForwardEngine, IDisposable
    {
        /// <summary>输出元素数超过该阈值才走多线程调度，小张量直接同步执行以免调度开销。</summary>
        private const int ParallelThreshold = 2048;
        private const int BatchSize = 64;

        private NativeArray<float> _data;     // 所有缓冲区共享的大数组
        private NativeArray<int> _bufOff;     // 每个缓冲区在 _data 中的起始偏移

        private NativeArray<byte> _op;
        private NativeArray<int> _in0, _in1, _in2, _outB;
        private NativeArray<int> _listOff, _list;
        private NativeArray<int> _bufRank, _shapeOff, _shapeData;
        private NativeArray<int> _wOff, _wRank, _wShapeOff, _wShapeData;
        private NativeArray<float> _wData;

        public NmpBurstEngine()
        {
            ValidateGraph();

            _op = new NativeArray<byte>(NmpIr.Op, Allocator.Persistent);
            _in0 = new NativeArray<int>(NmpIr.In0, Allocator.Persistent);
            _in1 = new NativeArray<int>(NmpIr.In1, Allocator.Persistent);
            _in2 = new NativeArray<int>(NmpIr.In2, Allocator.Persistent);
            _outB = new NativeArray<int>(NmpIr.Out, Allocator.Persistent);
            _listOff = new NativeArray<int>(NmpIr.ListOff, Allocator.Persistent);
            _list = new NativeArray<int>(NmpIr.List, Allocator.Persistent);
            _bufRank = new NativeArray<int>(NmpIr.BufferRank, Allocator.Persistent);
            _shapeOff = new NativeArray<int>(NmpIr.ShapeOff, Allocator.Persistent);
            _shapeData = new NativeArray<int>(NmpIr.ShapeData, Allocator.Persistent);
            _wOff = new NativeArray<int>(NmpIr.WeightOff, Allocator.Persistent);
            _wRank = new NativeArray<int>(NmpIr.WeightRank, Allocator.Persistent);
            _wShapeOff = new NativeArray<int>(NmpIr.WeightShapeOff, Allocator.Persistent);
            _wShapeData = new NativeArray<int>(NmpIr.WeightShapeData, Allocator.Persistent);
            _wData = new NativeArray<float>(NmpIr.WeightData, Allocator.Persistent);

            _bufOff = new NativeArray<int>(NmpIr.NBuffers + 1, Allocator.Persistent);
            int total = 0;
            for (int i = 0; i < NmpIr.NBuffers; i++)
            {
                _bufOff[i] = total;
                total += NmpIr.BufferCount[i];
            }
            _bufOff[NmpIr.NBuffers] = total;
            _data = new NativeArray<float>(total, Allocator.Persistent);
        }

        /// <summary>卷积内核只实现了 2 维、dilation=1、group=1，构造时先校验图是否满足。</summary>
        private static void ValidateGraph()
        {
            for (int i = 0; i < NmpIr.NNodes; i++)
            {
                if (NmpIr.Op[i] != NmpIr.Conv) continue;
                int p = NmpIr.ListOff[i];
                int r = NmpIr.List[p];
                if (r != 2) throw new NotSupportedException($"Burst 卷积仅支持 2 维空间，实际 {r}");
                for (int k = 0; k < r; k++)
                    if (NmpIr.List[p + 1 + 3 * r + k] != 1)
                        throw new NotSupportedException("Burst 卷积仅支持 dilation=1");
                if (NmpIr.List[p + 1 + 4 * r] != 1)
                    throw new NotSupportedException("Burst 卷积仅支持 group=1");
            }
        }

        public int InputLength => NmpIr.BufferCount[NmpIr.InputBuffer];

        public int BufferLength(int buffer) => NmpIr.BufferCount[buffer];

        /// <summary>执行一次前向计算。</summary>
        public void Run(float[] input)
        {
            int ib = NmpIr.InputBuffer;
            int n = NmpIr.BufferCount[ib];
            if (input.Length != n)
                throw new ArgumentException($"输入长度应为 {n}，实际为 {input.Length}");

            int baseOff = _bufOff[ib];
            for (int i = 0; i < n; i++) _data[baseOff + i] = input[i];

            var elem = MakeElementJob();
            var conv = MakeConvJob();

            for (int i = 0; i < NmpIr.NNodes; i++)
            {
                byte op = NmpIr.Op[i];
                if (op == NmpIr.Reshape || op == NmpIr.Unsqueeze)
                {
                    // 纯内存搬运，形状不同但元素顺序一致，无需 Burst
                    int srcB = NmpIr.In0[i], dstB = NmpIr.Out[i];
                    int src = _bufOff[srcB], dst = _bufOff[dstB];
                    int len = NmpIr.BufferCount[dstB];
                    for (int k = 0; k < len; k++) _data[dst + k] = _data[src + k];
                }
                else if (op == NmpIr.Conv)
                {
                    conv.node = i;
                    int count = NmpIr.BufferCount[NmpIr.Out[i]];
                    if (count >= ParallelThreshold) conv.Schedule(count, BatchSize).Complete();
                    else conv.Run(count);
                }
                else
                {
                    elem.node = i;
                    int count = NmpIr.BufferCount[NmpIr.Out[i]];
                    if (count >= ParallelThreshold) elem.Schedule(count, BatchSize).Complete();
                    else elem.Run(count);
                }
            }
        }

        /// <summary>把指定缓冲区的数据复制到 <paramref name="dst"/>。</summary>
        public void ReadBuffer(int buffer, float[] dst)
        {
            int off = _bufOff[buffer];
            int len = NmpIr.BufferCount[buffer];
            for (int i = 0; i < len; i++) dst[i] = _data[off + i];
        }

        private NmpElementJob MakeElementJob() => new NmpElementJob
        {
            op = _op,
            in0 = _in0,
            in1 = _in1,
            in2 = _in2,
            outB = _outB,
            listOff = _listOff,
            list = _list,
            bufRank = _bufRank,
            shapeOff = _shapeOff,
            shapeData = _shapeData,
            wRank = _wRank,
            wShapeOff = _wShapeOff,
            wShapeData = _wShapeData,
            wOff = _wOff,
            weight = _wData,
            bufOff = _bufOff,
            data = _data,
        };

        private NmpConvJob MakeConvJob() => new NmpConvJob
        {
            in0 = _in0,
            in1 = _in1,
            in2 = _in2,
            outB = _outB,
            listOff = _listOff,
            list = _list,
            shapeOff = _shapeOff,
            shapeData = _shapeData,
            wShapeOff = _wShapeOff,
            wShapeData = _wShapeData,
            wOff = _wOff,
            weight = _wData,
            bufOff = _bufOff,
            data = _data,
        };

        public void Dispose()
        {
            _op.Dispose();
            _in0.Dispose();
            _in1.Dispose();
            _in2.Dispose();
            _outB.Dispose();
            _listOff.Dispose();
            _list.Dispose();
            _bufRank.Dispose();
            _shapeOff.Dispose();
            _shapeData.Dispose();
            _wOff.Dispose();
            _wRank.Dispose();
            _wShapeOff.Dispose();
            _wShapeData.Dispose();
            _wData.Dispose();
            _bufOff.Dispose();
            _data.Dispose();
        }
    }

    /// <summary>
    /// 逐元素 / 布局 / 规约算子（除 Conv 与纯搬运外的全部节点）。
    /// <see cref="Execute"/> 只负责输出张量中的第 <c>o</c> 个元素，因此可任意并行。
    /// 索引映射与 <c>NmpEngine</c> 中对应算子保持一一对应。
    /// </summary>
    [BurstCompile]
    internal struct NmpElementJob : IJobParallelFor
    {
        private const int MaxRank = 5;

        [ReadOnly] public NativeArray<byte> op;
        [ReadOnly] public NativeArray<int> in0, in1, in2, outB;
        [ReadOnly] public NativeArray<int> listOff, list;
        [ReadOnly] public NativeArray<int> bufRank, shapeOff, shapeData;
        [ReadOnly] public NativeArray<int> wRank, wShapeOff, wShapeData, wOff;
        [ReadOnly] public NativeArray<float> weight;
        [ReadOnly] public NativeArray<int> bufOff;
        [NativeDisableParallelForRestriction] public NativeArray<float> data;
        public int node;

        public void Execute(int o)
        {
            switch (op[node])
            {
                case NmpIr.Transpose: Transpose(o); break;
                case NmpIr.Slice: Slice(o); break;
                case NmpIr.Pad: Pad(o); break;
                case NmpIr.Concat: Concat(o); break;
                case NmpIr.ReduceSum: Reduce(o, 0); break;
                case NmpIr.ReduceMin: Reduce(o, 1); break;
                case NmpIr.ReduceMax: Reduce(o, 2); break;
                case NmpIr.Neg: Unary(o, 0); break;
                case NmpIr.Relu: Unary(o, 1); break;
                case NmpIr.Sigmoid: Unary(o, 2); break;
                case NmpIr.Sqrt: Unary(o, 3); break;
                case NmpIr.Log: Unary(o, 4); break;
                case NmpIr.Mul: Binary(o, 0); break;
                case NmpIr.Add: Binary(o, 1); break;
                case NmpIr.Sub: Binary(o, 2); break;
                case NmpIr.Div: Binary(o, 3); break;
                case NmpIr.Equal: Binary(o, 4); break;
                case NmpIr.Where: Where(o); break;
            }
        }

        // ------------------------------------------------------------ 形状工具

        private float ReadAt(int enc, int i)
            => enc >= 0 ? data[bufOff[enc] + i] : weight[wOff[-enc - 2] + i];

        private int ShapeOf(int enc, Span<int> dst)
        {
            if (enc >= 0)
            {
                int off = shapeOff[enc];
                int r = bufRank[enc];
                for (int i = 0; i < r; i++) dst[i] = shapeData[off + i];
                return r;
            }
            else
            {
                int w = -enc - 2;
                int off = wShapeOff[w];
                int r = wRank[w];
                for (int i = 0; i < r; i++) dst[i] = wShapeData[off + i];
                return r;
            }
        }

        /// <summary>把输入轴映射到 outRank 维输出上的有效步长（广播轴为 0）。</summary>
        private void EffStrides(int enc, int outRank, Span<int> eff)
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

        /// <summary>行优先步长。</summary>
        private void StridesOf(int buf, int rank, Span<int> strides)
        {
            int off = shapeOff[buf];
            int s = 1;
            for (int k = rank - 1; k >= 0; k--)
            {
                strides[k] = s;
                s *= shapeData[off + k];
            }
        }

        private static int Reflect(int j, int n)
        {
            if (n <= 1) return 0;
            int period = 2 * (n - 1);
            int m = ((j % period) + period) % period;
            return m >= n ? period - m : m;
        }

        // ---------------------------------------------------------------- 算子

        private void Transpose(int o)
        {
            int inB = in0[node], oB = outB[node];
            int p = listOff[node];
            int r = list[p++];
            int osOff = shapeOff[oB];

            Span<int> inStrides = stackalloc int[MaxRank];
            StridesOf(inB, r, inStrides);

            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                int c = rem % dim;
                rem /= dim;
                si += c * inStrides[list[p + k]];
            }
            data[bufOff[oB] + o] = data[bufOff[inB] + si];
        }

        private void Slice(int o)
        {
            int inB = in0[node], oB = outB[node];
            int p = listOff[node];
            int nSpec = list[p++];
            int r = bufRank[inB];
            int osOff = shapeOff[oB];

            Span<int> inStrides = stackalloc int[MaxRank];
            StridesOf(inB, r, inStrides);

            Span<int> start = stackalloc int[MaxRank];
            Span<int> step = stackalloc int[MaxRank];
            for (int k = 0; k < r; k++) { start[k] = 0; step[k] = 1; }
            for (int s = 0; s < nSpec; s++)
            {
                int ax = list[p++];
                int st = list[p++];
                p++;                        // end 已归一化，复制时无需使用
                int sp = list[p++];
                start[ax] = st;
                step[ax] = sp;
            }

            int rem = o, si = 0;
            for (int k = r - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                int c = rem % dim;
                rem /= dim;
                si += (start[k] + c * step[k]) * inStrides[k];
            }
            data[bufOff[oB] + o] = data[bufOff[inB] + si];
        }

        private void Pad(int o)
        {
            int inB = in0[node], oB = outB[node];
            int p = listOff[node];
            int r = list[p++];
            int isOff = shapeOff[inB];
            int osOff = shapeOff[oB];
            int before = p;
            int mode = list[p + 2 * r];     // 0 constant / 1 reflect / 2 edge

            Span<int> inStrides = stackalloc int[MaxRank];
            StridesOf(inB, r, inStrides);

            int rem = o, si = 0;
            bool inside = true;
            for (int k = r - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                int c = rem % dim;
                rem /= dim;
                int inDim = shapeData[isOff + k];
                int j = c - list[before + k];
                if (mode == 1) j = Reflect(j, inDim);
                else if (mode == 2) j = j < 0 ? 0 : (j >= inDim ? inDim - 1 : j);
                else if (j < 0 || j >= inDim) { inside = false; break; }
                si += j * inStrides[k];
            }
            data[bufOff[oB] + o] = inside ? data[bufOff[inB] + si] : 0f;
        }

        private void Concat(int o)
        {
            int oB = outB[node];
            int p = listOff[node];
            int axis = list[p];
            int nIns = list[p + 1];
            int r = bufRank[oB];
            int osOff = shapeOff[oB];

            Span<int> coord = stackalloc int[MaxRank];
            int rem = o;
            for (int k = r - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                coord[k] = rem % dim;
                rem /= dim;
            }

            int c = coord[axis];
            int acc = 0;
            int chosen = list[p + 2 + nIns - 1];
            for (int i = 0; i < nIns; i++)
            {
                int ib = list[p + 2 + i];
                int seg = shapeData[shapeOff[ib] + axis];
                if (c < acc + seg) { chosen = ib; break; }
                acc += seg;
            }

            int rr = bufRank[chosen];
            Span<int> st = stackalloc int[MaxRank];
            StridesOf(chosen, rr, st);

            int si = (c - acc) * st[axis];
            for (int k = 0; k < r; k++)
                if (k != axis) si += coord[k] * st[k];

            data[bufOff[oB] + o] = data[bufOff[chosen] + si];
        }

        private void Reduce(int o, int kind)   // 0=sum 1=min 2=max
        {
            int inB = in0[node], oB = outB[node];
            int p = listOff[node];
            p++;                                 // keepdims：两种取值下索引公式均成立
            int nAx = list[p++];

            int inRank = bufRank[inB];
            int oRank = bufRank[oB];
            int isOff = shapeOff[inB];
            int osOff = shapeOff[oB];

            int reduced = 0;
            for (int i = 0; i < nAx; i++) reduced |= 1 << list[p + i];

            Span<int> inStrides = stackalloc int[MaxRank];
            StridesOf(inB, inRank, inStrides);

            Span<int> coord = stackalloc int[MaxRank];
            int rem = o;
            for (int k = oRank - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                coord[k] = rem % dim;
                rem /= dim;
            }

            Span<int> map = stackalloc int[MaxRank];
            int oa = 0, baseIdx = 0;
            for (int k = 0; k < inRank; k++)
            {
                if ((reduced & (1 << k)) != 0) { map[k] = -1; continue; }
                map[k] = oa;
                baseIdx += coord[oa] * inStrides[k];
                oa++;
            }

            int total = 1;
            for (int i = 0; i < nAx; i++) total *= shapeData[isOff + list[p + i]];

            float acc0 = kind == 0 ? 0f : (kind == 1 ? float.PositiveInfinity : float.NegativeInfinity);
            for (int t = 0; t < total; t++)
            {
                int rem2 = t, ii = baseIdx;
                for (int i = nAx - 1; i >= 0; i--)
                {
                    int ax = list[p + i];
                    int dim = shapeData[isOff + ax];
                    int cc = rem2 % dim;
                    rem2 /= dim;
                    ii += cc * inStrides[ax];
                }
                float v = data[bufOff[inB] + ii];
                if (kind == 0) acc0 += v;
                else if (kind == 1) { if (v < acc0) acc0 = v; }
                else { if (v > acc0) acc0 = v; }
            }
            data[bufOff[oB] + o] = acc0;
        }

        private void Unary(int o, int kind)    // 0=neg 1=relu 2=sigmoid 3=sqrt 4=log
        {
            float v = data[bufOff[in0[node]] + o];
            float r;
            switch (kind)
            {
                case 0: r = -v; break;
                case 1: r = math.max(v, 0f); break;
                case 2: r = 1f / (1f + math.exp(-v)); break;
                case 3: r = math.sqrt(v); break;
                default: r = math.log(v); break;
            }
            data[bufOff[outB[node]] + o] = r;
        }

        private void Binary(int o, int kind)   // 0=mul 1=add 2=sub 3=div 4=equal
        {
            int oB = outB[node];
            int oRank = bufRank[oB];
            int osOff = shapeOff[oB];
            int encA = in0[node], encB = in1[node];

            Span<int> eA = stackalloc int[MaxRank];
            Span<int> eB = stackalloc int[MaxRank];
            EffStrides(encA, oRank, eA);
            EffStrides(encB, oRank, eB);

            int rem = o, ia = 0, ib = 0;
            for (int k = oRank - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                int c = rem % dim;
                rem /= dim;
                ia += c * eA[k];
                ib += c * eB[k];
            }

            float x = ReadAt(encA, ia), y = ReadAt(encB, ib);
            float r;
            switch (kind)
            {
                case 0: r = x * y; break;
                case 1: r = x + y; break;
                case 2: r = x - y; break;
                case 3: r = x / y; break;
                default: r = x == y ? 1f : 0f; break;
            }
            data[bufOff[oB] + o] = r;
        }

        private void Where(int o)
        {
            int oB = outB[node];
            int oRank = bufRank[oB];
            int osOff = shapeOff[oB];
            int eC0 = in0[node], eX0 = in1[node], eY0 = in2[node];

            Span<int> eC = stackalloc int[MaxRank];
            Span<int> eX = stackalloc int[MaxRank];
            Span<int> eY = stackalloc int[MaxRank];
            EffStrides(eC0, oRank, eC);
            EffStrides(eX0, oRank, eX);
            EffStrides(eY0, oRank, eY);

            int rem = o, ic = 0, ix = 0, iy = 0;
            for (int k = oRank - 1; k >= 0; k--)
            {
                int dim = shapeData[osOff + k];
                int c = rem % dim;
                rem /= dim;
                ic += c * eC[k];
                ix += c * eX[k];
                iy += c * eY[k];
            }
            data[bufOff[oB] + o] = ReadAt(eC0, ic) != 0f ? ReadAt(eX0, ix) : ReadAt(eY0, iy);
        }
    }

    /// <summary>
    /// 卷积算子（本模型全部为 2 维、stride=(1,k)、dilation=1、group=1，可选逐输出通道 bias）。
    /// 本模型的 bias 由图中融合的 BatchNorm 折叠而来（常量张量），并非独立输入缓冲区。
    /// 每个输出元素独立完成 C×KH×KW 次乘加，按输出元素并行展开。
    /// </summary>
    [BurstCompile]
    internal struct NmpConvJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> in0, in1, in2, outB;
        [ReadOnly] public NativeArray<int> listOff, list;
        [ReadOnly] public NativeArray<int> shapeOff, shapeData;
        [ReadOnly] public NativeArray<int> wShapeOff, wShapeData, wOff;
        [ReadOnly] public NativeArray<float> weight;
        [ReadOnly] public NativeArray<int> bufOff;
        [NativeDisableParallelForRestriction] public NativeArray<float> data;
        public int node;

        public void Execute(int index)
        {
            int inB = in0[node], oB = outB[node];
            int p = listOff[node];
            int r = list[p++];                              // 空间维度数（恒为 2）
            int sH = list[p], sW = list[p + 1]; p += r;     // strides
            int pH = list[p], pW = list[p + 1]; p += r;     // pads（begin）

            int wIdx = -in1[node] - 2;
            int wsOff = wShapeOff[wIdx];
            int M = wShapeData[wsOff + 0];
            int C = wShapeData[wsOff + 1];
            int KH = wShapeData[wsOff + 2];
            int KW = wShapeData[wsOff + 3];

            int isOff = shapeOff[inB];
            int H = shapeData[isOff + 2];
            int W = shapeData[isOff + 3];
            int osOff = shapeOff[oB];
            int OH = shapeData[osOff + 2];
            int OW = shapeData[osOff + 3];

            int plane = OH * OW;
            int m = index / plane;
            int rem = index - m * plane;
            int oh = rem / OW;
            int ow = rem - oh * OW;

            int ih0 = oh * sH - pH;
            int iw0 = ow * sW - pW;

            int inBase = bufOff[inB];
            int wBase = wOff[wIdx];
            int bIdx = in2[node] <= -2 ? -in2[node] - 2 : -1;
            float acc = bIdx >= 0 ? weight[wOff[bIdx] + m] : 0f;

            for (int c = 0; c < C; c++)
            {
                int xc = inBase + c * H * W;
                int wc = wBase + (m * C + c) * KH * KW;
                for (int kh = 0; kh < KH; kh++)
                {
                    int ih = ih0 + kh;
                    if ((uint)ih >= (uint)H) continue;
                    int xr = xc + ih * W;
                    int wr = wc + kh * KW;
                    for (int kw = 0; kw < KW; kw++)
                    {
                        int iw = iw0 + kw;
                        if ((uint)iw >= (uint)W) continue;
                        acc += data[xr + iw] * weight[wr + kw];
                    }
                }
            }
            data[bufOff[oB] + index] = acc;
        }
    }
}