using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using BasicPitch;
using Madmom;
using Mt3;

namespace Mp3ToSheet;

public enum EngineKind { BasicPitch, Mt3, Madmom }

/// <summary>
/// 三种转谱/节拍引擎的统一运行器：解码 → 引擎分析 → 写输出，三段分别计时。
/// 不依赖 WPF，日志通过 <see cref="Action{String}"/> 回调输出。
/// </summary>
public sealed class EngineRunner
{
    private readonly Action<string> _log;
    private BasicPitchConverter? _bp;
    private Mt3Converter? _mt3;
    private MadmomAnalyzer? _madmom;

    /// <summary>Madmom 流式分块长度（秒）。0 = 不分块（与 madmom 一次性解码等价）。</summary>
    public double MadmomChunkSeconds { get; set; } = 10;

    /// <summary>
    /// Madmom 相邻块的重叠长度（秒）。用于给 BLSTM 后向预热 —— 块开头后向状态是重置的，
    /// 重叠太短会在块边界产生 spurious 拍点并让后续拍号整体错位。
    /// 实际取值为 min(该值, ChunkSeconds * 0.5)。
    /// </summary>
    public double MadmomOverlapSeconds { get; set; } = 3;

    public EngineRunner(Action<string> log) => _log = log;

    private void Log(string s) => _log?.Invoke(s);

    /// <summary>解析引擎名（用于 CLI 参数）。</summary>
    public static EngineKind Parse(string s) => s.ToLowerInvariant() switch
    {
        "mt3" => EngineKind.Mt3,
        "madmom" => EngineKind.Madmom,
        _ => EngineKind.BasicPitch,
    };

    /// <summary>预加载指定引擎的模型（把模型加载耗时排除在分析计时之外）。</summary>
    public void Preload(EngineKind kind)
    {
        var sw = Stopwatch.StartNew();
        switch (kind)
        {
            case EngineKind.BasicPitch:
                _bp ??= new BasicPitchConverter();
                break;
            case EngineKind.Mt3:
                _mt3 ??= new Mt3Converter(FindModelFile("mt3_encoder.onnx"), FindModelFile("mt3_decoder.onnx"));
                break;
            case EngineKind.Madmom:
                if (_madmom == null) ConfigureMadmom();
                break;
        }
        sw.Stop();
        Log($"      模型加载 {sw.Elapsed.TotalSeconds:F2}s");
    }

    /// <summary>跑完整流程。返回 (输出路径, 摘要)。</summary>
    public (string outPath, string summary) Run(string input, float threshold, EngineKind engine)
    {
        var swTotal = Stopwatch.StartNew();

        var sw = Stopwatch.StartNew();
        var (audio, srcRate) = AudioLoader.Load(input);
        sw.Stop();
        double dur = audio.Length / (double)AudioLoader.TargetRate;
        Log($"      解码: {srcRate}Hz → {AudioLoader.TargetRate}Hz, {dur:F2}s   耗时 {sw.Elapsed.TotalSeconds:F2}s" +
            $"（{dur / Math.Max(1e-9, sw.Elapsed.TotalSeconds):F1}x 实时）");

        string baseName = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input))!,
            Path.GetFileNameWithoutExtension(input));

        sw.Restart();
        var (outPath, summary, _) = RunAudio(audio, baseName, threshold, engine);
        sw.Stop();
        double analyze = sw.Elapsed.TotalSeconds;
        Log($"      分析: 耗时 {analyze:F2}s（{dur / Math.Max(1e-9, analyze):F1}x 实时）");

        swTotal.Stop();
        double total = swTotal.Elapsed.TotalSeconds;
        Log($"      合计: {total:F2}s（{dur / Math.Max(1e-9, total):F1}x 实时）  →  {summary}");
        return (outPath, summary);
    }

    /// <summary>
    /// 对已解码音频跑引擎（跳过解码）。供 Bench 复用同一份解码结果横向对比三引擎。
    /// 返回 (输出路径, 摘要, 分析耗时秒)。
    /// </summary>
    public (string outPath, string summary, double seconds) RunAudio(
        float[] audio, string baseName, float threshold, EngineKind engine)
    {
        var sw = Stopwatch.StartNew();
        var (outPath, summary) = engine switch
        {
            EngineKind.Mt3 => RunMt3(audio, baseName),
            EngineKind.Madmom => RunMadmom(audio, baseName),
            _ => RunBasicPitch(audio, baseName, threshold),
        };
        sw.Stop();
        return (outPath, summary, sw.Elapsed.TotalSeconds);
    }

    private (string, string) RunBasicPitch(float[] audio, string baseName, float threshold)
    {
        _bp ??= new BasicPitchConverter();
        var options = new NotesConvertOptions { FrameThreshold = threshold, OnsetThreshold = 0.5f };

        var notes = _bp.Convert(audio, AudioLoader.TargetRate, options);
        string midiPath = baseName + ".mid";
        // 注意：不要调 _bp.ConvertToMidi(...) —— 它内部会再跑一次 Convert（完整前向推理）。
        // 直接把已算出的 notes 交给 MidiWriter，避免推理两遍。
        new MidiWriter(notes).Write(midiPath, new MidiWriteOptions());
        return (midiPath, $"{notes.Count} 个音符");
    }

    private (string, string) RunMt3(float[] audio, string baseName)
    {
        // 用路径构造（而非 byte[]）：模型带外部数据文件 mt3_*.onnx.data，
        // 走路径 ONNX Runtime 才能定位到同目录的 .data。
        _mt3 ??= new Mt3Converter(FindModelFile("mt3_encoder.onnx"), FindModelFile("mt3_decoder.onnx"));

        var notes = _mt3.Convert(audio, AudioLoader.TargetRate, 1024);
        string midiPath = baseName + ".mt3.mid";
        Mt3MidiWriter.Write(notes, midiPath);

        var byInst = notes.GroupBy(n => n.IsDrum ? 9 : n.Program).OrderBy(g => g.Key);
        string inst = string.Join(", ", byInst.Select(g =>
            $"{(g.Key == 9 ? "Drums" : "P" + g.Key)}:{g.Count()}"));
        return (midiPath, $"{notes.Count} 个音符（{inst}）");
    }

    private (string, string) RunMadmom(float[] audio, string baseName)
    {
        ConfigureMadmom();
        var res = _madmom!.Analyze(audio, AudioLoader.TargetRate);
        if (res.Downbeats.Count == 0) throw new InvalidOperationException("未检出任何拍点");

        string outPath = baseName + ".beats.txt";
        var inv = CultureInfo.InvariantCulture;
        var lines = new List<string>
        {
            "# time_sec beat_in_bar beats_per_bar bpm",
            $"# engine=madmom fps={res.Fps.ToString(inv)}"
        };
        double sumBpm = 0; int nBpm = 0;
        var meterCount = new Dictionary<int, int>();
        foreach (var b in res.Downbeats)
        {
            lines.Add(string.Format(inv, "{0:F4} {1} {2} {3:F2}", b.Time, b.BeatInBar, b.BeatsPerBar, b.Bpm));
            meterCount[b.BeatsPerBar] = meterCount.TryGetValue(b.BeatsPerBar, out var c) ? c + 1 : 1;
            if (b.Bpm > 1 && b.Bpm < 400) { sumBpm += b.Bpm; nBpm++; }
        }
        foreach (var t in res.Beats) lines.Add(string.Format(inv, "BEAT {0:F4}", t));
        File.WriteAllLines(outPath, lines);

        int meter = meterCount.OrderByDescending(kv => kv.Value).First().Key;
        double avgBpm = nBpm > 0 ? sumBpm / nBpm : 0;
        return (outPath, $"拍号 {meter}/4，平均 BPM≈{avgBpm:F1}，{res.Downbeats.Count} 拍点" +
                         $"（beat 模型 {res.Beats.Count}）");
    }

    private void ConfigureMadmom()
    {
        if (_madmom == null)
            _madmom = new MadmomAnalyzer(MadmomDownBeatIr.Build(), MadmomBeatIr.Build());
        _madmom.ChunkSeconds = MadmomChunkSeconds;
        _madmom.OverlapSeconds = MadmomOverlapSeconds;
    }

    /// <summary>查找 ONNX 模型：优先 mt3_onnx/ 子目录，再沿目录向上找。</summary>
    private static string FindModelFile(string name)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            var inSub = Path.Combine(dir.FullName, "mt3_onnx", name);
            if (File.Exists(inSub)) return inSub;
            var flat = Path.Combine(dir.FullName, name);
            if (File.Exists(flat)) return flat;
        }
        throw new FileNotFoundException($"找不到模型文件 {name}（已在 mt3_onnx/ 与向上 8 层目录中查找）");
    }

    public void Dispose()
    {
        _bp?.Dispose();
        _mt3?.Dispose();
    }
}
