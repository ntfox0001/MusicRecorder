using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// madmom ml/hmm.pyx 的忠实 C# 移植：TransitionModel（CSR 稀疏）、ObservationModel、
/// HiddenMarkovModel + 对数域 Viterbi 解码。
///
/// 与 madmom 完全一致：
///  - TransitionModel 用类 CSR（compressed sparse row）表示：对目标状态 s，其入边来源状态
///    存于 states[pointers[s]..pointers[s+1]]，对应概率在 probabilities 同区间。
///  - ObservationModel 仅持有一个 pointers 数组（长度 = 状态数），把每个 HMM 状态映射到
///    log_densities 返回的 2D 密度数组的某一列。
///  - HiddenMarkovModel.viterbi 在对数域做 Viterbi：density = densities[frame, ptr[state]]，
///    对每个状态遍历其入边取最大 previous + logProb + density，回溯得到最优路径与 log 概率。
/// </summary>
public sealed class TransitionModel
{
    /// <summary>CSR 行内（入边来源状态）索引，uint32 对应物。</summary>
    public int[] States;

    /// <summary>CSR 行指针，长度 = num_states + 1。</summary>
    public int[] Pointers;

    /// <summary>转移概率（线性概率，非对数）。</summary>
    public double[] Probabilities;

    public int NumStates => Pointers.Length - 1;

    private double[]? _logProbabilities;

    /// <summary>转移对数概率（懒缓存）。</summary>
    public double[] LogProbabilities
    {
        get
        {
            if (_logProbabilities == null)
            {
                _logProbabilities = new double[Probabilities.Length];
                for (int i = 0; i < Probabilities.Length; i++)
                    _logProbabilities[i] = Math.Log(Probabilities[i] < 1e-300 ? 1e-300 : Probabilities[i]);
            }
            return _logProbabilities;
        }
    }

    public TransitionModel(int[] states, int[] pointers, double[] probabilities)
    {
        States = states;
        Pointers = pointers;
        Probabilities = probabilities;
    }

    /// <summary>
    /// 把稠密转移（dest[], src[], prob[]）压缩为 CSR，与 scipy.csr_matrix((prob,(dest,src)))
    /// 等价：按 dest 分组、对相同 (dest,src) 求和（消除重复）、组内按 src 升序排列（与 scipy 一致，
    /// 保证并列取最大时的 tie-break 顺序一致），概率分布校验（每个 src 出边和 ≈ 1）被放宽处理。
    /// </summary>
    public static (int[] States, int[] Pointers, double[] Probabilities) MakeSparse(
        int[] dest, int[] src, double[] prob)
    {
        int n = dest.Length;
        int numStates = 0;
        for (int i = 0; i < n; i++)
        {
            if (dest[i] > numStates) numStates = dest[i];
            if (src[i] > numStates) numStates = src[i];
        }
        numStates++; // max + 1

        // 按 dest 分组，组内按 src 聚合（消除重复坐标，scipy 会求和）
        var groups = new List<SortedDictionary<int, double>>(numStates);
        for (int d = 0; d < numStates; d++) groups.Add(new SortedDictionary<int, double>());
        for (int i = 0; i < n; i++)
        {
            var g = groups[dest[i]];
            if (g.TryGetValue(src[i], out double cur)) g[src[i]] = cur + prob[i];
            else g[src[i]] = prob[i];
        }

        var pointers = new int[numStates + 1];
        int count = 0;
        for (int d = 0; d < numStates; d++)
        {
            pointers[d] = count;
            count += groups[d].Count;
        }
        pointers[numStates] = count;

        var statesOut = new int[count];
        var probOut = new double[count];
        int k = 0;
        for (int d = 0; d < numStates; d++)
        {
            // SortedDictionary 已按 src 升序（与 scipy csr 排序一致）
            foreach (var kv in groups[d])
            {
                statesOut[k] = kv.Key;
                probOut[k] = kv.Value;
                k++;
            }
        }
        return (statesOut, pointers, probOut);
    }
}

/// <summary>
/// Viterbi 回溯矩阵。状态数 ≤ 65535 时用 ushort 存储（内存减半），否则回退 int。
/// 长音频下这是最大内存开销（frames × numStates），必须压缩。
/// </summary>
internal sealed class BacktrackBuffer
{
    private readonly ushort[]? _u16;
    private readonly int[]? _i32;

    public BacktrackBuffer(int frames, int numStates)
    {
        long cells = (long)frames * numStates;
        if (cells > int.MaxValue)
            throw new OutOfMemoryException($"Viterbi 回溯矩阵过大：{frames} × {numStates} = {cells} 单元，请减小分块长度。");
        if (numStates <= ushort.MaxValue) _u16 = new ushort[cells];
        else _i32 = new int[cells];
    }

    public void Set(long i, int v)
    {
        if (_u16 != null) _u16[i] = (ushort)v;
        else _i32![i] = v;
    }

    public int Get(long i) => _u16 != null ? _u16[i] : _i32![i];
}

/// <summary>
/// 观测模型基类：仅持 pointers（状态 → 密度列），具体密度由子类实现。
/// </summary>
public abstract class ObservationModel
{
    public int[] Pointers;

    protected ObservationModel(int[] pointers)
    {
        Pointers = pointers;
    }

    /// <summary>返回 (N, K) 对数密度；K 为密度列数。</summary>
    public abstract double[][] LogDensities(object observations);
}

/// <summary>
/// HiddenMarkovModel：持有转移模型 + 观测模型 + 初始分布，提供 Viterbi 解码。
/// </summary>
public sealed class HiddenMarkovModel
{
    private readonly TransitionModel _tm;
    private readonly ObservationModel _om;
    private readonly double[] _initial;

    public TransitionModel TransitionModel => _tm;
    public ObservationModel ObservationModel => _om;

    public HiddenMarkovModel(TransitionModel tm, ObservationModel om, double[]? initial = null)
    {
        _tm = tm;
        _om = om;
        int numStates = tm.NumStates;
        if (initial == null)
        {
            _initial = new double[numStates];
            double u = 1.0 / numStates;
            for (int i = 0; i < numStates; i++) _initial[i] = u;
        }
        else
        {
            _initial = (double[])initial.Clone();
        }
    }

    private const double NegInf = double.NegativeInfinity;

    /// <summary>
    /// 对数域 Viterbi。densities[f] 为第 f 帧的长度-K 数组；omPointers[state] ∈ [0,K)。
    /// 返回 (path, logProb)；若路径概率为 -inf 返回空 path（与 madmom 一致）。
    /// </summary>
    public (int[] Path, double LogProb) Viterbi(double[][] densities, int[] omPointers)
    {
        int numStates = _tm.NumStates;
        int N = densities.Length;
        if (N == 0) return (Array.Empty<int>(), NegInf);

        var tmStates = _tm.States;
        var tmPointers = _tm.Pointers;
        var tmLog = _tm.LogProbabilities;

        double[] prevVit = new double[numStates];
        for (int s = 0; s < numStates; s++) prevVit[s] = Math.Log(_initial[s] < 1e-300 ? 1e-300 : _initial[s]);

        var bt = new BacktrackBuffer(N, numStates);
        var cur = new double[numStates];

        for (int f = 0; f < N; f++)
        {
            var dens = densities[f];
            long rowBase = (long)f * numStates;
            for (int s = 0; s < numStates; s++)
            {
                cur[s] = NegInf;
                double density = dens[omPointers[s]];
                int p0 = tmPointers[s];
                int p1 = tmPointers[s + 1];
                int bestPrev = 0;
                for (int p = p0; p < p1; p++)
                {
                    int prev = tmStates[p];
                    double v = prevVit[prev] + tmLog[p] + density;
                    if (v > cur[s])
                    {
                        cur[s] = v;
                        bestPrev = prev;
                    }
                }
                bt.Set(rowBase + s, cur[s] > NegInf ? bestPrev : 0);
            }
            // previous = current
            var tmp = prevVit; prevVit = cur; cur = tmp;
        }

        // 末帧最优状态
        int state = 0;
        double best = NegInf;
        for (int s = 0; s < numStates; s++)
            if (prevVit[s] > best) { best = prevVit[s]; state = s; }

        double logProb = best;
        if (double.IsInfinity(logProb))
        {
            return (Array.Empty<int>(), logProb);
        }

        var path = new int[N];
        for (int f = N - 1; f >= 0; f--)
        {
            path[f] = state;
            state = bt.Get((long)f * numStates + state);
        }
        return (path, logProb);
    }
}
