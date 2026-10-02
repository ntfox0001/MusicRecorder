using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// madmom features/beats_hmm.py 的忠实 C# 移植：状态空间（Beat/Bar）、转移模型
/// （Beat/Bar TransitionModel）、观测模型（RNNBeat / RNNDownBeat）以及指数 tempo 转移。
/// 这是 beat / downbeat DBN 的底层数学，全部按源码实现，无启发式改写。
/// </summary>

/// <summary>Beat 状态空间（单拍）。</summary>
public sealed class BeatStateSpace
{
    public int[] Intervals;
    public int NumStates;
    public int NumIntervals;
    public int[] FirstStates;     // 每个间隔的首状态
    public int[] LastStates;      // 每个间隔的末状态
    public double[] StatePositions; // 每状态在拍内的归一化位置 [0,1)
    public int[] StateIntervals;    // 每状态对应的间隔（=该拍长度，帧数）

    public BeatStateSpace(double minInterval, double maxInterval, int? numIntervals = null)
    {
        // 默认线性间隔
        var lin = new List<int>();
        int lo = (int)Math.Round(minInterval);
        int hi = (int)Math.Round(maxInterval);
        for (int i = lo; i <= hi; i++) lin.Add(i);

        List<int> intervals;
        if (numIntervals.HasValue && numIntervals.Value < lin.Count)
        {
            // 对数间隔（base=2），迭代增加候选数直到唯一取整后数量达标
            int numLog = numIntervals.Value;
            double log2Min = Math.Log(minInterval, 2);
            double log2Max = Math.Log(maxInterval, 2);
            var uniq = new List<int>();
            while (uniq.Count < numIntervals.Value)
            {
                uniq.Clear();
                var vals = new double[numLog];
                for (int k = 0; k < numLog; k++)
                {
                    double t = numLog == 1 ? log2Min : log2Min + (log2Max - log2Min) * k / (numLog - 1);
                    vals[k] = Math.Pow(2, t);
                }
                // np.unique(np.round(...))：四舍五入到最近偶（与 numpy 一致），去重升序
                var rounded = new SortedSet<int>();
                foreach (var v in vals)
                    rounded.Add((int)Math.Round(v, MidpointRounding.ToEven));
                uniq.AddRange(rounded);
                numLog++;
            }
            intervals = uniq;
        }
        else
        {
            intervals = lin;
        }

        Intervals = intervals.ToArray();
        NumIntervals = Intervals.Length;
        NumStates = 0;
        foreach (var i in Intervals) NumStates += i;

        FirstStates = new int[NumIntervals];
        LastStates = new int[NumIntervals];
        StatePositions = new double[NumStates];
        StateIntervals = new int[NumStates];

        int idx = 0;
        int cum = 0;
        for (int b = 0; b < NumIntervals; b++)
        {
            int i = Intervals[b];
            FirstStates[b] = cum;
            LastStates[b] = cum + i - 1;
            for (int k = 0; k < i; k++)
            {
                StatePositions[idx] = (double)k / i; // linspace(0,1,i,endpoint=False)
                StateIntervals[idx] = i;
                idx++;
            }
            cum += i;
        }
    }
}

/// <summary>Bar 状态空间（多拍叠成一个小节）。</summary>
public sealed class BarStateSpace
{
    public int NumBeats;
    public double[] StatePositions; // 长度 num_states，范围 [0, num_beats)
    public int[] StateIntervals;
    public int NumStates;
    public int[] FirstStates;       // 全部拍的首状态（拼接成 1D，用于「移除所有首状态」）
    public int[] LastStates;        // 全部拍的末状态（拼接成 1D）
    /// <summary>每拍的首状态数组（长度 = num_intervals），与 madmom 的 first_states 列表一致。</summary>
    public int[][] FirstStatesByBeat;
    /// <summary>每拍的末状态数组（长度 = num_intervals），与 madmom 的 last_states 列表一致。</summary>
    public int[][] LastStatesByBeat;

    public BarStateSpace(int numBeats, double minInterval, double maxInterval, int? numIntervals = null)
    {
        NumBeats = numBeats;
        var bss = new BeatStateSpace(minInterval, maxInterval, numIntervals);

        var pos = new List<double>();
        var intr = new List<int>();
        var first = new List<int>();
        var last = new List<int>();
        FirstStatesByBeat = new int[numBeats][];
        LastStatesByBeat = new int[numBeats][];
        NumStates = 0;
        for (int b = 0; b < numBeats; b++)
        {
            var fb = new int[bss.FirstStates.Length];
            var lb = new int[bss.LastStates.Length];
            for (int k = 0; k < bss.StatePositions.Length; k++)
                pos.Add(bss.StatePositions[k] + b);
            for (int k = 0; k < bss.StateIntervals.Length; k++)
                intr.Add(bss.StateIntervals[k]);
            for (int k = 0; k < bss.FirstStates.Length; k++)
            {
                fb[k] = bss.FirstStates[k] + NumStates;
                first.Add(fb[k]);
            }
            for (int k = 0; k < bss.LastStates.Length; k++)
            {
                lb[k] = bss.LastStates[k] + NumStates;
                last.Add(lb[k]);
            }
            FirstStatesByBeat[b] = fb;
            LastStatesByBeat[b] = lb;
            NumStates += bss.NumStates;
        }
        StatePositions = pos.ToArray();
        StateIntervals = intr.ToArray();
        FirstStates = first.ToArray();
        LastStates = last.ToArray();
    }
}

/// <summary>指数 tempo 转移（beats_hmm.exponential_transition）。</summary>
internal static class HmmMath
{
    public static double Spacing1 => 2.220446049250313e-16; // np.spacing(1)

    public static double Clamp01(double x) => x < 0 ? 0 : (x > 1 ? 1 : x);
    public static double ClampPos(double x) => x < 1e-12 ? 1e-12 : x;

    /// <summary>
    /// ratio = to/from（逐元素），prob = exp(-lambda*|ratio-1|)，≤threshold 置 0，按行归一。
    /// 返回 (from_count x to_count) 矩阵。lambda=None 时返回单位阵（本实现恒假设非 null）。
    /// </summary>
    public static double[,] ExponentialTransition(int[] fromIntervals, int[] toIntervals, double transitionLambda)
    {
        int m = fromIntervals.Length;
        int n = toIntervals.Length;
        var prob = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            double fi = fromIntervals[i];
            double rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                double ratio = toIntervals[j] / fi;
                double p = Math.Exp(-transitionLambda * Math.Abs(ratio - 1.0));
                if (p <= Spacing1) p = 0;
                prob[i, j] = p;
                rowSum += p;
            }
            if (rowSum > 0)
            {
                for (int j = 0; j < n; j++) prob[i, j] /= rowSum;
            }
            else
            {
                // 退化保护：无有效转移时退化为均匀
                for (int j = 0; j < n; j++) prob[i, j] = 1.0 / n;
            }
        }
        return prob;
    }
}

/// <summary>Beat 转移模型（beats_hmm.BeatTransitionModel）。</summary>
public sealed class BeatTransitionModel
{
    public BeatStateSpace StateSpace;
    public TransitionModel Tm;

    public BeatTransitionModel(BeatStateSpace st, double transitionLambda)
    {
        StateSpace = st;
        int numStates = st.NumStates;

        // 同 tempo 转移：所有非首状态 s → s+1，概率 1
        var destSame = new List<int>();
        var srcSame = new List<int>();
        var probSame = new List<double>();
        var firstSet = new HashSet<int>(st.FirstStates);
        for (int s = 0; s < numStates; s++)
        {
            if (firstSet.Contains(s)) continue; // 首状态无同 tempo 入边
            destSame.Add(s);
            srcSame.Add(s - 1);
            probSame.Add(1.0);
        }

        // tempo 边界转移：末状态 → 首状态，指数分布
        int[] fromInt = new int[st.LastStates.Length];
        int[] toInt = new int[st.FirstStates.Length];
        for (int i = 0; i < st.LastStates.Length; i++) fromInt[i] = st.StateIntervals[st.LastStates[i]];
        for (int j = 0; j < st.FirstStates.Length; j++) toInt[j] = st.StateIntervals[st.FirstStates[j]];
        var exp = HmmMath.ExponentialTransition(fromInt, toInt, transitionLambda);

        var destT = new List<int>();
        var srcT = new List<int>();
        var probT = new List<double>();
        for (int i = 0; i < st.LastStates.Length; i++)
        {
            for (int j = 0; j < st.FirstStates.Length; j++)
            {
                double p = exp[i, j];
                if (p != 0)
                {
                    destT.Add(st.FirstStates[j]);
                    srcT.Add(st.LastStates[i]);
                    probT.Add(p);
                }
            }
        }

        int total = destSame.Count + destT.Count;
        var dest = new int[total];
        var src = new int[total];
        var prob = new double[total];
        int w = 0;
        for (int i = 0; i < destSame.Count; i++) { dest[w] = destSame[i]; src[w] = srcSame[i]; prob[w] = probSame[i]; w++; }
        for (int i = 0; i < destT.Count; i++) { dest[w] = destT[i]; src[w] = srcT[i]; prob[w] = probT[i]; w++; }

        var (states, pointers, probs) = TransitionModel.MakeSparse(dest, src, prob);
        Tm = new TransitionModel(states, pointers, probs);
    }
}

/// <summary>Bar 转移模型（beats_hmm.BarTransitionModel）。</summary>
public sealed class BarTransitionModel
{
    public BarStateSpace StateSpace;
    public TransitionModel Tm;

    public BarTransitionModel(BarStateSpace st, double transitionLambda)
    {
        StateSpace = st;
        int numStates = st.NumStates;
        var firstSet = new HashSet<int>(st.FirstStates);

        // 同 tempo 转移：每个拍内连续状态 s → s+1（首状态除外）
        var destSame = new List<int>();
        var srcSame = new List<int>();
        var probSame = new List<double>();
        for (int s = 0; s < numStates; s++)
        {
            if (firstSet.Contains(s)) continue;
            destSame.Add(s);
            srcSame.Add(s - 1);
            probSame.Add(1.0);
        }

        // 拍边界 tempo 转移：上一拍「各间隔末状态」→ 本拍「各间隔首状态」。
        // 注意 first_states[beat] / last_states[beat-1] 是长度 = num_intervals 的数组，
        // 因此每个拍边界是一个 num_intervals × num_intervals 的指数分布全交叉矩阵
        // （与 madmom 一致；beat=0 时上一拍取最后一拍，即小节循环边界）。
        var destT = new List<int>();
        var srcT = new List<int>();
        var probT = new List<double>();
        for (int beat = 0; beat < st.NumBeats; beat++)
        {
            int[] toStates = st.FirstStatesByBeat[beat];
            int[] fromStates = st.LastStatesByBeat[(beat - 1 + st.NumBeats) % st.NumBeats];
            int m = fromStates.Length, n = toStates.Length;
            var fromInt = new int[m];
            var toInt = new int[n];
            for (int i = 0; i < m; i++) fromInt[i] = st.StateIntervals[fromStates[i]];
            for (int j = 0; j < n; j++) toInt[j] = st.StateIntervals[toStates[j]];
            var exp = HmmMath.ExponentialTransition(fromInt, toInt, transitionLambda);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double p = exp[i, j];
                    if (p != 0)
                    {
                        destT.Add(toStates[j]);   // 目标：本拍某间隔首状态
                        srcT.Add(fromStates[i]);  // 来源：上一拍某间隔末状态
                        probT.Add(p);
                    }
                }
        }

        int total = destSame.Count + destT.Count;
        var dest = new int[total];
        var src = new int[total];
        var prob = new double[total];
        int w = 0;
        for (int i = 0; i < destSame.Count; i++) { dest[w] = destSame[i]; src[w] = srcSame[i]; prob[w] = probSame[i]; w++; }
        for (int i = 0; i < destT.Count; i++) { dest[w] = destT[i]; src[w] = srcT[i]; prob[w] = probT[i]; w++; }

        var (states, pointers, probs) = TransitionModel.MakeSparse(dest, src, prob);
        Tm = new TransitionModel(states, pointers, probs);
    }
}

/// <summary>Beat 观测模型（beats_hmm.RNNBeatTrackingObservationModel）。</summary>
public sealed class RNNBeatTrackingObservationModel : ObservationModel
{
    public BeatStateSpace StateSpace;
    private readonly int _observationLambda;

    public RNNBeatTrackingObservationModel(BeatStateSpace st, int observationLambda)
        : base(BuildPointers(st, observationLambda))
    {
        StateSpace = st;
        _observationLambda = observationLambda;
    }

    private static int[] BuildPointers(BeatStateSpace st, int observationLambda)
    {
        var ptr = new int[st.NumStates];
        double border = 1.0 / observationLambda;
        for (int s = 0; s < st.NumStates; s++)
            ptr[s] = st.StatePositions[s] < border ? 1 : 0;
        return ptr;
    }

    /// <summary>observations 为 1D beat 概率。返回 (N,2)：列0=非拍，列1=拍。</summary>
    public override double[][] LogDensities(object observations)
    {
        var obs = (double[])observations;
        int N = obs.Length;
        var dens = new double[N][];
        double denom = (_observationLambda - 1);
        for (int t = 0; t < N; t++)
        {
            double o = HmmMath.Clamp01(obs[t]);
            dens[t] = new[]
            {
                Math.Log(HmmMath.ClampPos((1.0 - o) / denom)),
                Math.Log(HmmMath.ClampPos(o))
            };
        }
        return dens;
    }
}

/// <summary>DownBeat 观测模型（beats_hmm.RNNDownBeatTrackingObservationModel）。</summary>
public sealed class RNNDownBeatTrackingObservationModel : ObservationModel
{
    public BarStateSpace StateSpace;
    private readonly int _observationLambda;

    public RNNDownBeatTrackingObservationModel(BarStateSpace st, int observationLambda)
        : base(BuildPointers(st, observationLambda))
    {
        StateSpace = st;
        _observationLambda = observationLambda;
    }

    private static int[] BuildPointers(BarStateSpace st, int observationLambda)
    {
        var ptr = new int[st.NumStates];
        double border = 1.0 / observationLambda;
        for (int s = 0; s < st.NumStates; s++)
        {
            // 拍内首段（position%1 < border）标记为 beat(1)
            ptr[s] = (st.StatePositions[s] % 1.0) < border ? 1 : 0;
            // 小节首拍（position < border）标记为 downbeat(2)
            if (st.StatePositions[s] < border) ptr[s] = 2;
        }
        return ptr;
    }

    /// <summary>observations 为 (N,2)：列0=beat，列1=downbeat。返回 (N,3)：0=非拍，1=拍，2=downbeat。</summary>
    public override double[][] LogDensities(object observations)
    {
        var obs = (double[][])observations;
        int N = obs.Length;
        var dens = new double[N][];
        double denom = (_observationLambda - 1);
        for (int t = 0; t < N; t++)
        {
            double beat = HmmMath.Clamp01(obs[t][0]);
            double down = HmmMath.Clamp01(obs[t][1]);
            dens[t] = new[]
            {
                Math.Log(HmmMath.ClampPos((1.0 - (beat + down)) / denom)),
                Math.Log(HmmMath.ClampPos(beat)),
                Math.Log(HmmMath.ClampPos(down))
            };
        }
        return dens;
    }
}

/// <summary>
/// DBNBeatTrackingProcessor 的忠实移植（beats.py）：单一 BeatStateSpace + BeatTransitionModel
/// + RNNBeatTrackingObservationModel + HMM，Viterbi + peak 对齐（correct）。
/// </summary>
public sealed class DBNBeatTrackingProcessor
{
    public double Fps;
    public bool Correct;
    public double Threshold;
    public BeatStateSpace St;
    public RNNBeatTrackingObservationModel Om;
    public HiddenMarkovModel Hmm;

    public DBNBeatTrackingProcessor(double fps,
        double minBpm = 55, double maxBpm = 215, int? numTempi = null,
        double transitionLambda = 100, int observationLambda = 16,
        double threshold = 0, bool correct = true)
    {
        Fps = fps;
        Correct = correct;
        Threshold = threshold;
        double minInterval = 60.0 * fps / maxBpm;
        double maxInterval = 60.0 * fps / minBpm;
        St = new BeatStateSpace(minInterval, maxInterval, numTempi);
        var tm = new BeatTransitionModel(St, transitionLambda);
        Om = new RNNBeatTrackingObservationModel(St, observationLambda);
        Hmm = new HiddenMarkovModel(tm.Tm, Om);
    }

    /// <summary>
    /// 分块长度（帧）。0 = 不分块，一次性 Viterbi（与 madmom 完全一致，但回溯矩阵占
    /// frames × numStates 内存）。长音频务必设置为有限值以约束峰值内存。
    /// </summary>
    public int ChunkFrames = 0;

    /// <summary>相邻块重叠帧数：重叠区由前一块（有完整上下文）提供结果，后一块的开头区丢弃。</summary>
    public int OverlapFrames = 0;

    /// <summary>activations：1D beat 概率（长度 N）。返回拍点时刻（秒）。</summary>
    public List<double> Process(double[] activations)
    {
        int first = 0;
        double[] act = activations;
        if (Threshold != 0)
        {
            int lo = -1, hi = -1;
            for (int i = 0; i < act.Length; i++)
            {
                if (act[i] >= Threshold) { if (lo < 0) lo = i; hi = i; }
            }
            if (lo < 0) return new List<double>();
            first = lo;
            int last = Math.Min(act.Length, hi + 1);
            var slice = new double[last - first];
            Array.Copy(act, first, slice, 0, slice.Length);
            act = slice;
        }
        int n = act.Length;
        if (n == 0) return new List<double>();

        // 短音频或禁用分块：走原路径，结果与 madmom 逐拍一致
        if (ChunkFrames <= 0 || n <= ChunkFrames)
            return DecodeRange(act, 0, n, first);

        int step = Math.Max(1, ChunkFrames - OverlapFrames);
        var beats = new List<double>();
        for (int start = 0; start < n; start += step)
        {
            int end = Math.Min(n, start + ChunkFrames);
            var sub = new double[end - start];
            Array.Copy(act, start, sub, 0, sub.Length);
            int acceptFrom = start == 0 ? 0 : OverlapFrames; // 丢弃与上一块重叠的开头区
            beats.AddRange(DecodeRange(sub, acceptFrom, sub.Length, first + start));
            if (end >= n) break;
        }
        return beats;
    }

    /// <summary>解码 sub 并按 peak 对齐收集 [acceptFrom, acceptTo) 内的拍点。</summary>
    private List<double> DecodeRange(double[] sub, int acceptFrom, int acceptTo, int offset)
    {
        var dens = Om.LogDensities(sub);
        var (path, _) = Hmm.Viterbi(dens, Om.Pointers);
        var beats = new List<double>();
        if (path.Length == 0) return beats;

        bool[] beatRange = new bool[path.Length];
        for (int i = 0; i < path.Length; i++) beatRange[i] = Om.Pointers[path[i]] == 1;

        if (Correct)
        {
            var idx = new List<int>();
            for (int i = 0; i + 1 < path.Length; i++)
                if (beatRange[i] != beatRange[i + 1]) idx.Add(i + 1);
            if (beatRange[0]) idx.Insert(0, 0);
            if (beatRange[^1]) idx.Add(beatRange.Length);
            for (int r = 0; r + 1 <= idx.Count; r += 2)
            {
                int left = idx[r], right = idx[r + 1];
                double best = -1; int peak = left;
                for (int f = left; f < right; f++)
                    if (sub[f] > best) { best = sub[f]; peak = f; }
                if (peak >= acceptFrom && peak < acceptTo) beats.Add((peak + offset) / Fps);
            }
        }
        else
        {
            for (int i = 1; i < path.Length; i++)
                if (!beatRange[i - 1] && beatRange[i] && i >= acceptFrom && i < acceptTo)
                    beats.Add((i + offset) / Fps);
        }
        return beats;
    }
}

/// <summary>
/// DBNDownBeatTrackingProcessor 的忠实移植（downbeats.py）：对每个候选 beats_per_bar 各建一个独立
/// HMM（BarStateSpace + BarTransitionModel + RNNDownBeatTrackingObservationModel），并行 Viterbi，
/// 选 log 概率最高者；correct 时按 peak 对齐，返回 (time_sec, beat_number)。
/// </summary>
public sealed class DBNDownBeatTrackingProcessor
{
    public double Fps;
    public bool Correct;
    public double Threshold;
    public int[] BeatsPerBar;
    public List<HiddenMarkovModel> Hmms = new();
    public List<RNNDownBeatTrackingObservationModel> Oms = new();

    public DBNDownBeatTrackingProcessor(int[] beatsPerBar, double fps,
        double minBpm = 55, double maxBpm = 215, int numTempi = 60,
        double transitionLambda = 100, int observationLambda = 16,
        double threshold = 0.05, bool correct = true)
    {
        Fps = fps;
        Correct = correct;
        Threshold = threshold;
        BeatsPerBar = (int[])beatsPerBar.Clone();
        double minInterval = 60.0 * fps / maxBpm;
        double maxInterval = 60.0 * fps / minBpm;
        for (int b = 0; b < BeatsPerBar.Length; b++)
        {
            var st = new BarStateSpace(BeatsPerBar[b], minInterval, maxInterval, numTempi);
            var tm = new BarTransitionModel(st, transitionLambda);
            var om = new RNNDownBeatTrackingObservationModel(st, observationLambda);
            Hmms.Add(new HiddenMarkovModel(tm.Tm, om));
            Oms.Add(om);
        }
    }

    public sealed class Result
    {
        public double[] Times = Array.Empty<double>();   // 拍点时刻（秒）
        public double[] BeatNumbers = Array.Empty<double>(); // 拍号（1-based，小节内）
        public int BeatsPerBar;                            // 被选中的拍号
    }

    /// <summary>
    /// 分块长度（帧）。0 = 不分块，一次性 Viterbi（与 madmom 完全一致，但每个 pattern 都要一块
    /// frames × numStates 的回溯矩阵）。长音频务必设置有限值以约束峰值内存。
    /// </summary>
    public int ChunkFrames = 0;

    /// <summary>相邻块重叠帧数：重叠区由前一块（有完整上下文）提供结果，后一块的开头区丢弃。</summary>
    public int OverlapFrames = 0;

    /// <summary>activations：(N,2)，列0=beat，列1=downbeat。返回拍点与拍号。</summary>
    public Result Process(double[][] activations)
    {
        int first = 0;
        double[][] act = activations;
        if (Threshold != 0)
        {
            int lo = -1, hi = -1;
            for (int i = 0; i < act.Length; i++)
            {
                if (act[i][0] >= Threshold || act[i][1] >= Threshold)
                {
                    if (lo < 0) lo = i; hi = i;
                }
            }
            if (lo < 0) return new Result();
            first = lo;
            int last = Math.Min(act.Length, hi + 1);
            var slice = new double[last - first][];
            for (int i = 0; i < slice.Length; i++) slice[i] = act[first + i];
            act = slice;
        }
        int n = act.Length;
        if (n == 0) return new Result();

        // 切块（不分块时等价于单块，结果与原实现逐拍一致）
        int chunk = ChunkFrames > 0 ? ChunkFrames : n;
        int step = ChunkFrames > 0 ? Math.Max(1, ChunkFrames - OverlapFrames) : n;
        var starts = new List<int>();
        for (int s = 0; s < n; s += step)
        {
            starts.Add(s);
            if (s + chunk >= n) break;
        }

        // 第一遍：每块对每个 pattern 解码，累加 log 概率（拍号判定保持全局）
        var logSum = new double[Hmms.Count];
        var chunkBeats = new List<(int[] Frames, int[] Nums)[]>();
        for (int c = 0; c < starts.Count; c++)
        {
            int s = starts[c];
            int e = Math.Min(n, s + chunk);
            var sub = new double[e - s][];
            for (int i = 0; i < sub.Length; i++) sub[i] = act[s + i];

            var perPattern = new (int[], int[])[Hmms.Count];
            for (int h = 0; h < Hmms.Count; h++)
            {
                var dens = Oms[h].LogDensities(sub);
                var (path, logp) = Hmms[h].Viterbi(dens, Oms[h].Pointers);
                if (!double.IsInfinity(logp)) logSum[h] += logp;
                Extract(path, sub, Oms[h], out var fr, out var nu);
                perPattern[h] = (fr.ToArray(), nu.ToArray());
            }
            chunkBeats.Add(perPattern);
        }

        int bestIdx = 0;
        for (int h = 1; h < Hmms.Count; h++) if (logSum[h] > logSum[bestIdx]) bestIdx = h;
        int bpb = BeatsPerBar[bestIdx];

        // 第二遍：按全局最佳 pattern 拼接。采纳区首尾相接（块0 取 [0,L)，其余块丢弃开头
        // OverlapFrames），故拍序列连续。
        //
        // 拍号相位：每块沿用**自身解码**的相对相位，只在块边界对齐一次（用重叠区内最后一个
        // 拍点算出相位偏移 delta）。若采用「跨块盲目 +1 递增」，一旦某块多检/漏检一拍，
        // 错位会永久累积到后续所有块 —— 实测拍号一致率仅 ~40%。改为逐块对齐后误差不跨块传播。
        var times = new List<double>();
        var nums = new List<double>();
        for (int c = 0; c < starts.Count; c++)
        {
            var (fr, nu) = chunkBeats[c][bestIdx];
            int acceptFrom = c == 0 ? 0 : OverlapFrames;

            int firstK = -1;
            for (int k = 0; k < fr.Length; k++)
                if (fr[k] >= acceptFrom) { firstK = k; break; }
            if (firstK < 0) continue;

            int delta = 0;
            if (c > 0)
            {
                // 重叠区（被丢弃的开头）内最后一个拍点的编号作为锚点
                int anchor = -1;
                for (int k = 0; k < firstK; k++) anchor = nu[k];
                if (anchor >= 0)
                {
                    int expected = anchor % bpb + 1;          // 下一拍应有的编号
                    delta = expected - nu[firstK];            // 本块相对相位的整体补偿
                }
            }

            for (int k = firstK; k < fr.Length; k++)
            {
                int m = (nu[k] + delta - 1) % bpb;
                if (m < 0) m += bpb;
                times.Add((fr[k] + starts[c] + first) / Fps);
                nums.Add(m + 1);
            }
        }
        return new Result { Times = times.ToArray(), BeatNumbers = nums.ToArray(), BeatsPerBar = bpb };
    }

    /// <summary>从 path 提取拍点帧与拍号（correct 时按 peak 对齐）。</summary>
    private void Extract(int[] path, double[][] sub, RNNDownBeatTrackingObservationModel om,
        out List<int> frames, out List<int> nums)
    {
        frames = new List<int>();
        nums = new List<int>();
        if (path.Length == 0) return;

        var st = om.StateSpace;
        var beatNumbers = new int[path.Length];
        for (int i = 0; i < path.Length; i++)
            beatNumbers[i] = (int)st.StatePositions[path[i]] + 1;

        if (Correct)
        {
            bool[] beatRange = new bool[path.Length];
            for (int i = 0; i < path.Length; i++) beatRange[i] = om.Pointers[path[i]] >= 1;
            var idx = new List<int>();
            for (int i = 0; i + 1 < path.Length; i++)
                if (beatRange[i] != beatRange[i + 1]) idx.Add(i + 1);
            if (beatRange[0]) idx.Insert(0, 0);
            if (beatRange[^1]) idx.Add(beatRange.Length);
            for (int r = 0; r + 1 <= idx.Count; r += 2)
            {
                int left = idx[r], right = idx[r + 1];
                // 取 beat/downbeat 中最大激活的帧（等价于 numpy argmax 扁平 //2）
                double best = -1; int peak = left;
                for (int f = left; f < right; f++)
                {
                    double score = Math.Max(sub[f][0], sub[f][1]);
                    if (score > best) { best = score; peak = f; }
                }
                frames.Add(peak);
                nums.Add(beatNumbers[peak]);
            }
        }
        else
        {
            for (int i = 1; i < path.Length; i++)
                if (beatNumbers[i] != beatNumbers[i - 1])
                {
                    frames.Add(i);
                    nums.Add(beatNumbers[i]);
                }
        }
    }
}
