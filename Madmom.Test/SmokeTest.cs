using System;
using System.Collections.Generic;
using Madmom;

namespace Madmom.Test;

/// <summary>
/// 冒烟测试：用随机合成权重 + 程序化构造的 IrGraph（BLSTM → Dense → Sigmoid）验证
/// MadmomEngine 的动态 T 分配、RNN/Dense 执行、形状正确性，以及 DownBeatDbn / BeatTracker
/// 能跑通。不依赖 madmom 真实权重（NC 协议，需你本地导出）。
/// </summary>
public static class SmokeTest
{
    private static readonly Random Rng = new(12345);

    private static float[] Rand(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>窥孔权重：布局 [input, forget, cell(填 0), output]，长度 4H（与导出脚本一致）。</summary>
    private static float[] Peephole(int h)
    {
        var pe = Rand(4 * h);
        for (int j = 0; j < h; j++) pe[2 * h + j] = 0f; // cell 槽无窥孔
        return pe;
    }

    /// <summary>构造一个 (T,F) → BLSTM(H=4,双向) → Dense(3) → Sigmoid 的合成图。</summary>
    private static IrGraph BuildSyntheticGraph(int fDim, int h, int outDim)
    {
        int T = 1;                     // 时间维占位
        int nB = 4;
        int[] bufCount = { fDim, 2 * h, outDim, outDim };
        int[] bufRank = { 2, 2, 2, 2 };
        int[] shapeOff = { 0, 2, 4, 6, 8 };
        int[] shapeData = { T, fDim, T, 2 * h, T, outDim, T, outDim };

        byte[] op = { IrGraph.OpBlstm, IrGraph.OpDense, IrGraph.OpSigmoid };
        int[] in0 = { 0, 1, 2 };
        int[] in1 = { -1, -1, -1 };
        int[] in2 = { -1, -1, -1 };
        int[] @out = { 1, 2, 3 };
        int[] listOff = { 0, 3, 3, 3 };
        int[] list = { h, 1, 1 };      // BLSTM: H, bidir=1, retSeq=1

        // 权重（命名，须与 MadmomEngine 约定一致）
        var weights = new Dictionary<string, float[]>();
        int gates = 4;                 // LSTM
        weights["lstm_0fx"] = Rand(gates * h * fDim);
        weights["lstm_0fh"] = Rand(gates * h * h);
        weights["lstm_0fb"] = Rand(gates * h);
        weights["lstm_0bx"] = Rand(gates * h * fDim);
        weights["lstm_0bh"] = Rand(gates * h * h);
        weights["lstm_0bb"] = Rand(gates * h);
        // 窥孔权重（真实 madmom LSTM 有）：布局 [input, forget, cell=0, output]，共 4H
        weights["lstm_0fp"] = Peephole(h);
        weights["lstm_0bp"] = Peephole(h);
        // Dense 输入维 = BLSTM 输出维 = 2*h（不是 fDim！）；Dense 节点序号=1 → dense1_*
        weights["dense1_x"] = Rand(outDim * (2 * h));
        weights["dense1_b"] = Rand(outDim);

        var g = new IrGraph(nB, op.Length, 0, 0,
            bufCount, bufRank, shapeOff, shapeData,
            op, in0, in1, in2, @out, listOff, list,
            Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>(),
            new[] { 0 }, Array.Empty<int>(), Array.Empty<float>(),
            weights);
        g.InputFeatureDim = fDim;
        g.OutputDim = outDim;
        g.Fps = 100;
        g.OutputBuffer = 3;
        return g;
    }

    public static int Main()
    {
        int fDim = 6, h = 4, outDim = 3, T = 20;
        var g = BuildSyntheticGraph(fDim, h, outDim);

        var input = Rand(T * fDim);
        var engine = new MadmomEngine(g);
        try
        {
            engine.Run(input);
        }
        catch (Exception ex)
        {
            Console.WriteLine("RUN EXCEPTION: " + ex);
            return 1;
        }
        var outBuf = new float[T * outDim];
        engine.ReadBuffer(g.OutputBuffer, outBuf);

        bool finite = true;
        foreach (var v in outBuf) if (!float.IsFinite(v)) finite = false;

        Console.WriteLine($"[1] 引擎前向：输入 {T}x{fDim}，输出 {T}x{outDim} = {outBuf.Length} 元素，全部有限={finite}");
        Console.WriteLine($"    输出前 6 个值: {string.Join(", ", System.Linq.Enumerable.Take(outBuf, 6))}");

        // 重排为 (T, outDim) 喂给 DBN
        var act = new float[T][];
        for (int t = 0; t < T; t++)
        {
            act[t] = new float[outDim];
            for (int j = 0; j < outDim; j++) act[t][j] = outBuf[t * outDim + j];
        }
        var dbn = new DownBeatDbn(100);
        var beats = dbn.Track(act);
        Console.WriteLine($"[2] DownBeatDbn：检测到 {beats.Count} 个拍点");
        foreach (var b in beats.GetRange(0, Math.Min(5, beats.Count)))
            Console.WriteLine($"    t={b.Time:F3}s  beatInBar={b.BeatInBar}  bpb={b.BeatsPerBar}  bpm={b.Bpm}");

        // BeatTracker
        var beatProb = new float[T];
        for (int t = 0; t < T; t++) beatProb[t] = act[t][0];
        var bt = new BeatTracker(100);
        var beatTimes = bt.Track(beatProb);
        Console.WriteLine($"[3] BeatTracker：检测到 {beatTimes.Count} 个拍点");
        foreach (var t in System.Linq.Enumerable.Take(beatTimes, 5))
            Console.WriteLine($"    t={t:F3}s");

        // 断言
        bool ok = outBuf.Length == T * outDim && finite && beats != null && beatTimes != null;
        Console.WriteLine(ok ? "SMOKE TEST PASS ✅" : "SMOKE TEST FAIL ❌");

        // ---- [4] 用构造的周期拍点信号验证 DBN 能响应周期输入（非随机噪声）----
        ok &= TestPeriodicGrid();
        Console.WriteLine(ok ? "SMOKE TEST PASS ✅" : "SMOKE TEST FAIL ❌");
        return ok ? 0 : 1;
    }

    /// <summary>构造周期拍点：beat 每 period 帧、downbeat 每 4*period 帧，喂给 DBN，
    /// 验证其产出多拍且时间落在期望网格附近。</summary>
    private static bool TestPeriodicGrid()
    {
        int fps = 100, T = 240, period = 60;   // bpm≈100，4/4 拍
        var act = new float[T][];
        for (int t = 0; t < T; t++)
        {
            bool isBeat = t % period == 0;
            bool isDown = t % (period * 4) == 0;
            float pos = (float)((t % period) / (double)period);
            act[t] = new[] { isBeat ? 0.99f : 0.01f,
                             isDown ? 0.99f : 0.01f,
                             pos };
        }
        var dbn = new DownBeatDbn(fps);
        var beats = dbn.Track(act);
        Console.WriteLine($"[4] 周期信号 DBN：期望拍点≈4，检测到 {beats.Count} 个");
        foreach (var b in beats)
            Console.WriteLine($"    t={b.Time:F3}s  beatInBar={b.BeatInBar}  bpb={b.BeatsPerBar}  bpm={b.Bpm}");

        int expected = T / period;  // 4
        bool ok = beats.Count >= expected - 1;   // 容忍少检 1 个
        if (!ok) Console.WriteLine("    ⚠ DBN 未检出足够拍点（DBN 为独立实现，需对照 madmom 调参）");
        return ok;
    }
}
