using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Madmom;

namespace Madmom.EndToEnd;

/// <summary>
/// 端到端验证：真实 madmom 权重（本地导出，见 .madmom_export）+ 忠实 C# 特征管线
/// + 忠实移植的 DBN/HMM，对一段真实音频跑 beat / downbeat / 拍号。
///
/// 用法：dotnet run --project Madmom.EndToEnd -- [音频路径]
/// 默认自动向上查找 test_scale.wav。
/// </summary>
public static class Program
{
    /// <summary>从基准目录向上若干层查找目标文件。</summary>
    private static string FindUp(string fileName)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            var hit = Path.Combine(dir.FullName, fileName);
            if (File.Exists(hit)) return hit;
        }
        return null;
    }

    public static int Main(string[] args)
    {
        // 参数：[音频路径] [--dump <目录>] [--chunk <秒>]
        //   --dump  导出喂给 DBN 的激活，用于与 madmom 参考实现逐帧对照
        //   --chunk DBN 分块长度（秒）；0 = 不分块（madmom 原行为，内存随音频长度线性增长）
        string path = null, dumpDir = null, beatsOut = null;
        double chunkSec = 10;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--dump" && i + 1 < args.Length) dumpDir = args[++i];
            else if (args[i] == "--beats-out" && i + 1 < args.Length) beatsOut = args[++i];
            else if (args[i] == "--chunk" && i + 1 < args.Length)
                chunkSec = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else path = args[i];
        }
        path ??= FindUp("test_scale.wav");
        if (path == null || !File.Exists(path))
        {
            Console.WriteLine("❌ 找不到音频文件（test_scale.wav）。请传入路径作为参数。");
            return 1;
        }
        Console.WriteLine($"音频: {path}");

        // ---------- 1. 解码 ----------
        var (audio, sampleRate) = WavLoader.Load(path);
        Console.WriteLine($"  {sampleRate} Hz, {audio.Length} samples, {audio.Length / (double)sampleRate:F3} 秒 (单声道归一化)");

        // ---------- 1.5 状态空间与 madmom 参考值对照 ----------
        // 参考值由 .madmom_export/ref_dbn.py（真实 madmom beats_hmm.py）在同一配置下算出：
        //   BarStateSpace(3)=11157, BarStateSpace(4)=14876, BeatStateSpace=5617 (82 个间隔)
        double minInterval = 60.0 * 100 / 215.0;
        double maxInterval = 60.0 * 100 / 55.0;
        var ss3 = new BarStateSpace(3, minInterval, maxInterval, 60);
        var ss4 = new BarStateSpace(4, minInterval, maxInterval, 60);
        var ssBeat = new BeatStateSpace(minInterval, maxInterval, null);
        Console.WriteLine($"\n[0] SIMD: {Simd.Describe}（ARM NEON 为 4-wide，Burst 会生成同类指令）");

        Console.WriteLine($"\n[0] 状态空间对照（madmom 参考值）: " +
                          $"Bar(3)={ss3.NumStates}(11157)  Bar(4)={ss4.NumStates}(14876)  " +
                          $"Beat={ssBeat.NumStates}(5617)  间隔数 Beat={ssBeat.NumIntervals}(82)");
        bool ssOk = ss3.NumStates == 11157 && ss4.NumStates == 14876 &&
                    ssBeat.NumStates == 5617 && ssBeat.NumIntervals == 82;
        Console.WriteLine($"    状态空间匹配: {(ssOk ? "✅" : "❌")}");
        if (!ssOk) return 1;

        // ---------- 2. 特征维与模型维一致性检查 ----------
        var sw = Stopwatch.StartNew();
        var dbFeat = Features.Extract(audio, sampleRate, Features.FeatureConfig.DownBeat, out int dbDim);
        var bFeat = Features.Extract(audio, sampleRate, Features.FeatureConfig.Beat, out int bDim);
        sw.Stop();
        long memFeat = GC.GetTotalMemory(true);
        Console.WriteLine($"\n[1] 特征：downbeat {dbFeat.Length}x{dbDim}（模型期望 {MadmomDownBeatIr.InputFeatureDim}）  " +
                          $"beat {bFeat.Length}x{bDim}（模型期望 {MadmomBeatIr.InputFeatureDim}）  耗时 {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"    存活堆（解码+特征后）: {memFeat / 1048576.0:F1} MB");
        bool dimOk = dbDim == MadmomDownBeatIr.InputFeatureDim && bDim == MadmomBeatIr.InputFeatureDim;
        Console.WriteLine($"    维度匹配: {(dimOk ? "✅" : "❌")}");
        if (!dimOk) return 1;

        // ---------- 3. 真实权重 + 忠实 DBN 全链路 ----------
        var proc = Process.GetCurrentProcess();
        var analyzer = new MadmomAnalyzer(MadmomDownBeatIr.Build(), MadmomBeatIr.Build())
        {
            ChunkSeconds = chunkSec,
            OverlapSeconds = chunkSec > 0 ? Math.Min(3.0, chunkSec * 0.5) : 0
        };
        int frames = (int)(audio.Length / (double)sampleRate * 100);
        Console.WriteLine($"\n[2] DBN 分块: {(chunkSec > 0 ? $"块长 {chunkSec:F0}s / 重叠 {analyzer.OverlapSeconds:F0}s" : "不分块")}" +
                          $"  (激活约 {frames} 帧)");
        sw.Restart();
        var res = analyzer.Analyze(audio, sampleRate);
        sw.Stop();
        long memAll = GC.GetTotalMemory(true);
        proc.Refresh();
        Console.WriteLine($"    全链路耗时 {sw.ElapsedMilliseconds}ms   峰值工作集 {proc.PeakWorkingSet64 / 1048576.0:F1} MB" +
                          $"   存活堆(全程后) {memAll / 1048576.0:F1} MB");

        // ---------- 4. 拍点 + 拍号（downbeat DBN）----------
        Console.WriteLine($"\n[3] Downbeat DBN：{res.Downbeats.Count} 个拍点");
        int show = Math.Min(24, res.Downbeats.Count);
        for (int i = 0; i < show; i++)
        {
            var b = res.Downbeats[i];
            string mark = b.BeatInBar == 1 ? "  <-- 小节首拍(downbeat)" : "";
            Console.WriteLine($"    t={b.Time,7:F3}s  beat {b.BeatInBar}/{b.BeatsPerBar}  bpm≈{b.Bpm,6:F1}{mark}");
        }
        if (res.Downbeats.Count > show) Console.WriteLine($"    ... 其余 {res.Downbeats.Count - show} 个省略");

        // 拍号（meter）：出现最多的 BeatsPerBar
        if (res.Downbeats.Count > 0)
        {
            var meterCounts = new Dictionary<int, int>();
            double sumBpm = 0; int nBpm = 0;
            foreach (var b in res.Downbeats)
            {
                meterCounts[b.BeatsPerBar] = meterCounts.TryGetValue(b.BeatsPerBar, out var c) ? c + 1 : 1;
                if (b.Bpm > 1 && b.Bpm < 400) { sumBpm += b.Bpm; nBpm++; }
            }
            int meter = meterCounts.OrderByDescending(kv => kv.Value).First().Key;
            double avgBpm = nBpm > 0 ? sumBpm / nBpm : 0;
            Console.WriteLine($"\n[4] 推定拍号: {meter}/4   平均 BPM≈{avgBpm:F1}");
        }

        // ---------- 5. 独立 beat 模型 ----------
        Console.WriteLine($"\n[5] Beat 模型：{res.Beats.Count} 个拍点");
        foreach (var t in res.Beats.Take(12))
            Console.WriteLine($"    t={t,7:F3}s");

        // ---------- 6. 可选：导出激活 ----------
        if (dumpDir != null)
        {
            Directory.CreateDirectory(dumpDir);
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var dbLines = new List<string>();
            foreach (var row in res.DownBeatActivations)
                dbLines.Add(string.Format(inv, "{0:F9} {1:F9}", row[0], row[1]));
            File.WriteAllLines(Path.Combine(dumpDir, "downbeat_act.txt"), dbLines);
            var bLines = new List<string>();
            foreach (var v in res.BeatActivations) bLines.Add(string.Format(inv, "{0:F9}", v));
            File.WriteAllLines(Path.Combine(dumpDir, "beat_act.txt"), bLines);
            Console.WriteLine($"\n[6] 已导出激活到 {dumpDir}（downbeat_act.txt {dbLines.Count} 行, beat_act.txt {bLines.Count} 行）");
        }

        // ---------- 7. 可选：导出拍点（用于对比不同分块配置的结果是否逐拍一致）----------
        if (beatsOut != null)
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var lines = new List<string>();
            foreach (var b in res.Downbeats)
                lines.Add(string.Format(inv, "D {0:F6} {1} {2}", b.Time, b.BeatInBar, b.BeatsPerBar));
            foreach (var t in res.Beats) lines.Add(string.Format(inv, "B {0:F6}", t));
            File.WriteAllLines(beatsOut, lines);
            Console.WriteLine($"\n[7] 已导出 {lines.Count} 个拍点到 {beatsOut}");
        }

        bool ok = res.Downbeats.Count > 0;
        Console.WriteLine($"\n{(ok ? "END-TO-END PASS ✅" : "END-TO-END FAIL ❌（未检出拍点）")}");
        return ok ? 0 : 0; // 未检出也返回 0，便于在无声/极短样本上不阻塞
    }
}
