using System;
using System.IO;
using System.Linq;
using BasicPitch;
using Mt3;

if (args.Length < 1)
{
    Console.WriteLine("用法:");
    Console.WriteLine("  BasicPitch (默认): BasicPitchSharp.Test <input> [output.mid]");
    Console.WriteLine("  MT3 模式:         BasicPitchSharp.Test --mt3 <input> [output.mid]");
    return;
}

bool useMt3 = args[0] == "--mt3";
if (useMt3) args = args.Skip(1).ToArray();

string input = args[0];
string output = args.Length > 1 ? args[1] : Path.ChangeExtension(input, useMt3 ? "_mt3.mid" : ".mid");

var audioReader = new NAudioAudioReader();

if (useMt3)
{
    RunMt3(audioReader, input, output);
}
else
{
    RunBasicPitch(audioReader, input, output);
}

void RunBasicPitch(BasicPitch.IAudioReader reader, string input, string output)
{
    Console.WriteLine($"读取音频: {input}");
    var (samples, sampleRate) = reader.Read(input);
    Console.WriteLine($"  采样率={sampleRate}, 采样数={samples.Length}, 时长={samples.Length / (float)sampleRate:F2}s");

    Console.WriteLine("初始化 Basic Pitch 模型...");
    using var converter = new BasicPitchConverter();
    Console.WriteLine("运行音高检测...");
    var notes = converter.Convert(samples, sampleRate);
    Console.WriteLine($"检测到 {notes.Count} 个音符:");
    foreach (var n in notes)
        Console.WriteLine($"  音高={n.Pitch} 开始={n.StartTime:F3}s 结束={n.EndTime:F3}s");
    converter.ConvertToMidi(samples, sampleRate, output);
    Console.WriteLine($"写入 MIDI: {output}");
}

void RunMt3(Mt3.IAudioReader reader, string input, string output)
{
    Console.WriteLine($"读取音频: {input}");
    var (samples, sampleRate) = reader.Read(input);
    Console.WriteLine($"  采样率={sampleRate}, 采样数={samples.Length}, 时长={samples.Length / (float)sampleRate:F2}s");

    string encPath = File.Exists(Path.Combine(AppContext.BaseDirectory, "mt3_encoder.onnx"))
        ? Path.Combine(AppContext.BaseDirectory, "mt3_encoder.onnx")
        : "mt3_encoder.onnx";
    string decPath = encPath.Replace("mt3_encoder", "mt3_decoder");

    Console.WriteLine($"\n初始化 MT3 模型 (byte[] 加载)...");
    Console.WriteLine($"  Encoder: {encPath}");
    // 模拟 Unity: 通过 UnityWebRequest 读取 ONNX 得到 byte[] 后传入构造函数
    byte[] encBytes = File.ReadAllBytes(encPath);
    byte[] decBytes = File.ReadAllBytes(decPath);
    using var converter = new Mt3Converter(encBytes, decBytes);
    Console.WriteLine("运行 MT3 转谱...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var progress = new Progress<double>(p =>
    {
        int pct = (int)(p * 100);
        Console.Write($"\r  进度: [{new string('#', pct / 2)}{new string('-', 50 - pct / 2)}] {pct}%");
    });
    var notes = converter.Convert(samples, sampleRate, 1024, progress);
    sw.Stop();
    Console.WriteLine();
    Console.WriteLine($"完成！耗时 {sw.Elapsed.TotalSeconds:F1}s，检测到 {notes.Count} 个音符:");

    var byProg = notes.GroupBy(n => n.IsDrum ? 9 : n.Program).OrderBy(g => g.Key);
    foreach (var g in byProg)
        Console.WriteLine($"  [{(g.Key == 9 ? "Drums" : $"Program {g.Key}")}] {g.Count()} notes");

    int shown = 0;
    foreach (var n in notes.OrderBy(n => n.StartTime))
    {
        if (shown++ >= 20) break;
        string inst = n.IsDrum ? "Drum" : $"P{n.Program}";
        Console.WriteLine($"    {inst} pitch={n.Pitch} start={n.StartTime:F3} end={n.EndTime:F3} vel={n.Velocity}");
    }

    Mt3MidiWriter.Write(notes, output);
    Console.WriteLine($"\n写入 MIDI: {output}");
}
