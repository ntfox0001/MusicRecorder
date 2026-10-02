using System.Diagnostics;
using System.Globalization;
using Mp3ToSheet;

namespace Mp3ToSheet;

/// <summary>
/// 三种引擎（BasicPitch / MT3 / Madmom）的命令行测速器。
/// 解码只做一次，三引擎共用同一份音频，保证横向可比。
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string? path = null;
        string enginesSpec = "all";
        double chunk = 10;
        double overlap = 3;
        float threshold = 0.3f;
        string? outDir = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--engine" when i + 1 < args.Length: enginesSpec = args[++i]; break;
                case "--chunk" when i + 1 < args.Length:
                    chunk = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--overlap" when i + 1 < args.Length:
                    overlap = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--threshold" when i + 1 < args.Length:
                    threshold = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--out" when i + 1 < args.Length: outDir = args[++i]; break;
                default: path ??= args[i]; break;
            }
        }

        if (path == null || !File.Exists(path))
        {
            Console.WriteLine("用法: Mp3ToSheet.Bench <音频文件> [--engine all|basicpitch|mt3|madmom|a,b,c]");
            Console.WriteLine("                                  [--chunk 秒] [--overlap 秒] [--threshold 0.3] [--out 输出目录]");
            Console.WriteLine();
            Console.WriteLine("--chunk / --overlap 仅对 Madmom 生效：流式分块长度与相邻块重叠（秒）。");
            Console.WriteLine("  chunk=0 关闭分块（等价 madmom 一次性解码，精度最高但内存随时长线性增长）。");
            Console.WriteLine("  重叠用于给 BLSTM 后向预热：太短会在块边界产生 spurious 拍点并使拍号错位。");
            return 1;
        }

        var engines = enginesSpec.Equals("all", StringComparison.OrdinalIgnoreCase)
            ? new[] { EngineKind.BasicPitch, EngineKind.Mt3, EngineKind.Madmom }
            : enginesSpec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                         .Select(EngineRunner.Parse).ToArray();

        var proc = Process.GetCurrentProcess();
        string nosimd = Environment.GetEnvironmentVariable("MADMOM_NOSIMD");
        Console.WriteLine($"=== Mp3ToSheet.Bench ===");
        Console.WriteLine($"音频: {Path.GetFileName(path)}   ({new FileInfo(path).Length / 1048576.0:F2} MB)");
        Console.WriteLine($"引擎: {string.Join(", ", engines)}    Madmom 分块: " +
                          $"{(chunk <= 0 ? "关闭" : $"{chunk}s / 重叠 {Math.Min(overlap, chunk * 0.5)}s")}");
        Console.WriteLine($"环境: {Environment.ProcessorCount} 核  .NET {Environment.Version}" +
                          $"  MADMOM_NOSIMD={(string.IsNullOrEmpty(nosimd) ? "未设置(启用SIMD)" : nosimd)}");
        Console.WriteLine();

        // ---------- 解码（只做一次） ----------
        var sw = Stopwatch.StartNew();
        var (audio, srcRate) = AudioLoader.Load(path);
        sw.Stop();
        double dur = audio.Length / (double)AudioLoader.TargetRate;
        double decodeSec = sw.Elapsed.TotalSeconds;
        Console.WriteLine($"[解码] {srcRate}Hz → {AudioLoader.TargetRate}Hz  时长 {dur:F2}s  " +
                          $"耗时 {decodeSec:F2}s（{dur / Math.Max(1e-9, decodeSec):F1}x 实时）");
        Console.WriteLine();

        string baseName = Path.GetFileNameWithoutExtension(path);
        string dir = outDir ?? Path.GetDirectoryName(Path.GetFullPath(path))!;
        if (outDir != null) Directory.CreateDirectory(outDir);

        var runner = new EngineRunner(Console.WriteLine)
        {
            MadmomChunkSeconds = chunk,
            MadmomOverlapSeconds = overlap,
        };

        var rows = new List<Row>();
        bool firstEngine = true;
        foreach (var e in engines)
        {
            Console.WriteLine($"----- {e} -----");
            string outPath = "-", summary = "(失败)";
            double loadSec = 0, analyzeSec = 0;
            try
            {
                var swLoad = Stopwatch.StartNew();
                runner.Preload(e);
                swLoad.Stop();
                loadSec = swLoad.Elapsed.TotalSeconds;

                var r = runner.RunAudio(audio, Path.Combine(dir, baseName), threshold, e);
                outPath = r.outPath;
                summary = r.summary;
                analyzeSec = r.seconds;
            }
            catch (Exception ex)
            {
                summary = "[异常] " + ex.Message;
                Console.WriteLine("  " + summary);
            }
            rows.Add(new Row(e, loadSec, analyzeSec, decodeSec, dur, summary, outPath)
            { IsFirst = firstEngine });
            firstEngine = false;
            Console.WriteLine();
        }
        runner.Dispose();

        // ---------- 汇总 ----------
        const string h1 = "引擎", h2 = "模型加载", h3 = "分析", h4 = "合计", h5 = "实时率";
        Console.WriteLine("==================== 汇总 ====================");
        Console.WriteLine($"{h1,-12} {h2,8} {h3,8} {h4,8} {h5,8}   结果");
        Console.WriteLine(new string('-', 96));
        foreach (var r in rows)
        {
            double total = r.Load + r.Analyze + DecodeShare(r, decodeSec);
            Console.WriteLine($"{r.Engine.ToString(),-12} {r.Load,8:F2} {r.Analyze,8:F2} {total,8:F2} " +
                              $"{r.Duration / Math.Max(1e-9, total),8:F1}x   {Trunc(r.Summary, 44)}");
        }
        Console.WriteLine(new string('-', 96));
        Console.WriteLine($"合计 = 模型加载 + 分析 + 解码({decodeSec:F2}s)；实时率 = 音频时长 / 合计");
        Console.WriteLine($"输出目录: {dir}");

        proc.Refresh();
        Console.WriteLine($"峰值工作集 {proc.PeakWorkingSet64 / 1048576.0:F0} MB   " +
                          $"存活堆 {GC.GetTotalMemory(false) / 1048576.0:F1} MB");
        return rows.Any(r => r.Summary.StartsWith("[异常]")) ? 2 : 0;
    }

    /// <summary>解码耗时分摊：只计入第一个引擎（其余复用同一份解码结果）。</summary>
    private static double DecodeShare(Row r, double decodeSec) => r.IsFirst ? decodeSec : 0;

    private static string Trunc(string s, int n) => s.Length <= n ? s : s[..(n - 1)] + "…";

    private sealed record Row(
        EngineKind Engine, double Load, double Analyze, double DecodeOnly,
        double Duration, string Summary, string OutPath)
    {
        public bool IsFirst { get; set; }
    }
}
