using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// 拍点 + 拍号（downbeat / meter）DBN 解码器。
///
/// 这是对 madmom features/downbeats.py 的 DBNDownBeatTrackingProcessor 的忠实封装：
/// 对每个候选 beats_per_bar 各建一个独立 HMM（BarStateSpace + BarTransitionModel +
/// RNNDownBeatTrackingObservationModel），并行 Viterbi 选最佳 log 概率，correct 时按 peak 对齐。
/// 默认参数（MIN_BPM=55, MAX_BPM=215, NUM_TEMPI=60, TRANSITION_LAMBDA=100,
/// OBSERVATION_LAMBDA=16, THRESHOLD=0.05, CORRECT=True, beats_per_bar=[3,4]）与 madmom 一致。
///
/// 输入激活形状 (T,2)：列0=beat 概率，列1=downbeat 概率。
/// </summary>
public sealed class DownBeatDbn
{
    private readonly DBNDownBeatTrackingProcessor _proc;

    public DownBeatDbn(double fps, int tempoMin = 55, int tempoMax = 215,
        int[]? beatsPerBar = null, double tempoJumpCost = 0.5)
    {
        int[] bpb = beatsPerBar ?? new[] { 3, 4 };
        // tempoJumpCost 为旧 API 兼容项；madmom 用固定 TRANSITION_LAMBDA=100
        _proc = new DBNDownBeatTrackingProcessor(bpb, fps,
            minBpm: tempoMin, maxBpm: tempoMax,
            numTempi: 60, transitionLambda: 100,
            observationLambda: 16, threshold: 0.05, correct: true);
    }

    /// <summary>底层 processor，可微调分块（ChunkFrames/OverlapFrames）等高级参数。</summary>
    public DBNDownBeatTrackingProcessor Processor => _proc;

    /// <summary>
    /// 设置分块解码（单位：帧）。长音频必设，否则回溯矩阵为 帧数 × 状态数 × 2B。
    /// </summary>
    public void SetChunking(int chunkFrames, int overlapFrames)
    {
        _proc.ChunkFrames = chunkFrames;
        _proc.OverlapFrames = overlapFrames;
    }

    public struct Beat { public double Time; public int BeatInBar; public int BeatsPerBar; public double Bpm; }

    public List<Beat> Track(float[][] activation)
    {
        int T = activation.Length;
        if (T == 0) return new List<Beat>();
        int cols = activation[0].Length;
        var act = new double[T][];
        for (int t = 0; t < T; t++)
        {
            act[t] = new[] { (double)activation[t][0], cols > 1 ? (double)activation[t][1] : 0.0 };
        }

        var res = _proc.Process(act);
        var outBeats = new List<Beat>();
        for (int i = 0; i < res.Times.Length; i++)
        {
            double bpm = 0;
            if (i > 0 && res.Times[i] > res.Times[i - 1])
                bpm = 60.0 / (res.Times[i] - res.Times[i - 1]);
            else if (i < res.Times.Length - 1 && res.Times[i + 1] > res.Times[i])
                bpm = 60.0 / (res.Times[i + 1] - res.Times[i]);
            outBeats.Add(new Beat
            {
                Time = res.Times[i],
                BeatInBar = (int)res.BeatNumbers[i],
                BeatsPerBar = res.BeatsPerBar,
                Bpm = bpm
            });
        }
        return outBeats;
    }
}
