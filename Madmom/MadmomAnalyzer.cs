using System;
using System.Diagnostics;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// 高层入口：音频 → 特征 → downbeat/beat 模型前向 → DBN → 拍点网格 + 拍号。
/// 与 BasicPitchConverter 对称：构造时注入模型（IrGraph，由 MadmomIr.Build() 生成），
/// Analyze 跑完整链路。
///
/// 真实模型权重由你本地用 export_madmom_ir.py 从 madmom 导出（见 README 许可说明，NC 协议）。
/// 特征提取（Features.cs）目前给出 madmom 风格谱通量的结构等价实现；要做到与 madmom
/// 数值逐帧一致，需把 madmom 的 SignalProcessor 配方也原样移植（见 README）。
/// </summary>
public sealed class MadmomAnalyzer
{
    private readonly IrGraph _downbeatModel;
    private readonly IrGraph? _beatModel;

    /// <summary>
    /// DBN 分块长度（秒）。Viterbi 回溯矩阵是 帧数 × 状态数 × 2B，长音频下是内存瓶颈。
    /// 分块后峰值内存与音频长度无关（只与块长有关）；短于块长的音频不分块，结果完全一致。
    /// 设为 0 可关闭分块（等价 madmom 一次性解码）。默认 10 秒。
    /// </summary>
    public double ChunkSeconds { get; set; } = 10;

    /// <summary>相邻块重叠秒数：重叠区结果取前一块（上下文完整），后一块开头丢弃。默认 3 秒。</summary>
    public double OverlapSeconds { get; set; } = 3;

    public MadmomAnalyzer(IrGraph downbeatModel, IrGraph? beatModel = null)
    {
        _downbeatModel = downbeatModel;
        _beatModel = beatModel;
        FeatureEngine = Seams.FeatureEngineFactory?.Invoke();   // Unity: Burst 引擎；桌面: null → 托管
    }

    /// <summary>特征引擎（构造时从 Seams 工厂取；可手动替换）。null = 托管 Features 实现。</summary>
    public IFeatureEngine? FeatureEngine { get; set; }

    public sealed class Result
    {
        public double Fps = 100;
        public List<DownBeatDbn.Beat> Downbeats = new();
        public List<double> Beats = new();

        /// <summary>喂给 downbeat DBN 的激活 (T,2)：列0=beat，列1=downbeat（便于与 madmom 对照）。</summary>
        public float[][] DownBeatActivations = Array.Empty<float[]>();

        /// <summary>喂给 beat DBN 的激活 (T,)：beat 概率。</summary>
        public float[] BeatActivations = Array.Empty<float>();

        /// <summary>上一次 Analyze 的阶段耗时（毫秒）：特征提取 / RNN 前向 / DownBeat DBN / Beat HMM / 合计。
        /// 常开（开销可忽略），供宿主（如 Unity 编辑器面板）显示性能分解。</summary>
        public double FeatureMs, RnnMs, DbnMs, BeatMs, TotalMs;
    }

    public Result Analyze(float[] audio, int sampleRate)
    {
        bool prof = Environment.GetEnvironmentVariable("MADMOM_PROFILE") == "1";
        var swTotal = Stopwatch.StartNew();
        double tFeat = 0, tRnn = 0, tDbn = 0, tBeat = 0;
        var swStage = new Stopwatch();

        double fps = _downbeatModel.Fps > 0 ? _downbeatModel.Fps : 100.0;

        int totalFrames = Features.FrameCount(audio, sampleRate);
        if (totalFrames == 0) return new Result { Fps = fps };

        // 流式块长：短于块长的音频不分块（结果与全量逐帧一致）
        int chunk = ChunkSeconds > 0 ? Math.Max(1, (int)Math.Round(ChunkSeconds * fps)) : totalFrames;
        bool streaming = chunk < totalFrames;

        // ---- downbeat：分块特征 → 分块前向 → 累积激活 ----
        // 激活只有 T×2 个 float（192 秒音频约 150 KB），累积它不影响常量内存目标；
        // 而特征矩阵与引擎缓冲都是块级的，故峰值内存与音频总长无关。
        var dbEngine = new MadmomEngine(_downbeatModel) { KeepState = streaming };
        dbEngine.ResetState();
        int dbOutDim = _downbeatModel.OutputDim > 0 ? _downbeatModel.OutputDim : 3;
        int keep = Math.Min(2, dbOutDim);
        int beatCol = Math.Min(1, dbOutDim - 1);     // 列 1 = beat
        int downCol = Math.Min(2, dbOutDim - 1);     // 列 2 = downbeat
        var act = new float[totalFrames][];

        for (int s = 0; s < totalFrames; s += chunk)
        {
            int cnt = Math.Min(chunk, totalFrames - s);
            swStage?.Restart();
            int dbFeatDim;
            var feat = FeatureEngine != null
                ? FeatureEngine.ExtractBlock(audio, sampleRate,
                    Features.FeatureConfig.DownBeat, s, cnt, out dbFeatDim)
                : Features.ExtractBlock(audio, sampleRate,
                    Features.FeatureConfig.DownBeat, s, cnt, out dbFeatDim);
            tFeat += swStage.Elapsed.TotalMilliseconds;

            // 构造模型输入 (cnt, dbFeatDim)
            var dbInput = new float[cnt * dbFeatDim];
            for (int t = 0; t < cnt; t++)
                for (int j = 0; j < dbFeatDim; j++)
                    dbInput[t * dbFeatDim + j] = feat[t][j];

            swStage?.Restart();
            dbEngine.Run(dbInput);
            tRnn += swStage.Elapsed.TotalMilliseconds;
            var dbOutBuf = new float[cnt * dbOutDim];
            dbEngine.ReadBuffer(_downbeatModel.OutputBuffer, dbOutBuf);

            // downbeat 模型输出 3 列 [non-beat, beat, downbeat]；madmom 用 np.delete(obj=0)
            // 丢弃第 0 列（non-beat）后，喂给 DBN 的是 [beat, downbeat] = 列 1,2。
            for (int t = 0; t < cnt; t++)
            {
                act[s + t] = new float[keep];
                act[s + t][0] = dbOutBuf[t * dbOutDim + beatCol]; // beat
                act[s + t][1] = dbOutBuf[t * dbOutDim + downCol]; // downbeat
            }
        }

        var dbn = new DownBeatDbn(fps);
        if (ChunkSeconds > 0)
            dbn.SetChunking((int)Math.Round(ChunkSeconds * fps),
                            (int)Math.Round(Math.Min(OverlapSeconds, ChunkSeconds * 0.5) * fps));
        swStage?.Restart();
        var downbeats = dbn.Track(act);
        tDbn += swStage.Elapsed.TotalMilliseconds;

        var result = new Result { Fps = fps, Downbeats = downbeats, DownBeatActivations = act };

        // ---- 可选：独立 beat 模型（使用 beat 特征配置，取其第 1 列 beat）----
        if (_beatModel != null)
        {
            var bEngine = new MadmomEngine(_beatModel) { KeepState = streaming };
            bEngine.ResetState();
            int bOutDim = _beatModel.OutputDim > 0 ? _beatModel.OutputDim : 2;
            var beatProb = new float[totalFrames];
            for (int s = 0; s < totalFrames; s += chunk)
            {
                int cnt = Math.Min(chunk, totalFrames - s);
                swStage?.Restart();
                int bFeatDim;
                var bFeat = FeatureEngine != null
                    ? FeatureEngine.ExtractBlock(audio, sampleRate,
                        Features.FeatureConfig.Beat, s, cnt, out bFeatDim)
                    : Features.ExtractBlock(audio, sampleRate,
                        Features.FeatureConfig.Beat, s, cnt, out bFeatDim);
                tFeat += swStage.Elapsed.TotalMilliseconds;
                var bInput = new float[cnt * bFeatDim];
                for (int t = 0; t < cnt; t++)
                    for (int j = 0; j < bFeatDim; j++)
                        bInput[t * bFeatDim + j] = bFeat[t][j];
                swStage?.Restart();
                bEngine.Run(bInput);
                tRnn += swStage.Elapsed.TotalMilliseconds;
                var bOutBuf = new float[cnt * bOutDim];
                bEngine.ReadBuffer(_beatModel.OutputBuffer, bOutBuf);
                for (int t = 0; t < cnt; t++) beatProb[s + t] = bOutBuf[t * bOutDim]; // 第一列 = beat
            }
            var bt = new BeatTracker(fps);
            if (ChunkSeconds > 0)
                bt.SetChunking((int)Math.Round(ChunkSeconds * fps),
                               (int)Math.Round(Math.Min(OverlapSeconds, ChunkSeconds * 0.5) * fps));
            swStage?.Restart();
            result.Beats = bt.Track(beatProb);
            tBeat += swStage.Elapsed.TotalMilliseconds;
            result.BeatActivations = beatProb;
        }

        result.FeatureMs = tFeat;
        result.RnnMs = tRnn;
        result.DbnMs = tDbn;
        result.BeatMs = tBeat;
        result.TotalMs = swTotal.Elapsed.TotalMilliseconds;

        if (prof)
        {
            Console.Error.WriteLine(
                $"[madmom-profile] stage feat={tFeat:F0}ms rnn={tRnn:F0}ms dbn={tDbn:F0}ms beatHMM={tBeat:F0}ms total={result.TotalMs:F0}ms");
        }

        return result;
    }
}
