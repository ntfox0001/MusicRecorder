using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// 仅 beat（不含拍号）的 DBN 追踪器，对应 madmom features/beats.py 的 DBNBeatTrackingProcessor。
///
/// 忠实封装：单一 BeatStateSpace + BeatTransitionModel + RNNBeatTrackingObservationModel +
/// HiddenMarkovModel，Viterbi + peak 对齐。默认参数（MIN_BPM=55, MAX_BPM=215, NUM_TEMPI=None 线性,
/// TRANSITION_LAMBDA=100, OBSERVATION_LAMBDA=16, THRESHOLD=0, CORRECT=True）与 madmom 一致。
///
/// 输入：beat 激活 (T,) 或 (T,1)，输出拍点时刻（秒）。
/// </summary>
public sealed class BeatTracker
{
    private readonly DBNBeatTrackingProcessor _proc;

    public BeatTracker(double fps, int tempoMin = 55, int tempoMax = 200, double tempoJumpCost = 0.3)
    {
        _proc = new DBNBeatTrackingProcessor(fps,
            minBpm: tempoMin, maxBpm: tempoMax,
            numTempi: null, transitionLambda: 100,
            observationLambda: 16, threshold: 0, correct: true);
    }

    /// <summary>底层 processor，可微调分块（ChunkFrames/OverlapFrames）等高级参数。</summary>
    public DBNBeatTrackingProcessor Processor => _proc;

    /// <summary>设置分块解码（单位：帧）。长音频必设，否则回溯矩阵为 帧数 × 状态数 × 2B。</summary>
    public void SetChunking(int chunkFrames, int overlapFrames)
    {
        _proc.ChunkFrames = chunkFrames;
        _proc.OverlapFrames = overlapFrames;
    }

    public List<double> Track(float[] beatProb)
    {
        int T = beatProb.Length;
        if (T == 0) return new List<double>();
        var act = new double[T];
        for (int t = 0; t < T; t++) act[t] = beatProb[t];
        return _proc.Process(act);
    }
}
