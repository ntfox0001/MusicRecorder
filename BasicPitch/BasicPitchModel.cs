using System;
using System.Collections.Generic;

namespace BasicPitch;

/// <summary>
/// Basic Pitch 模型推理。
///
/// 默认采用内置纯 C# 前向引擎（<see cref="NmpEngine"/>），模型参数已固化进 NmpIr.g.cs，
/// 不依赖 ONNX Runtime 或任何原生库，可在 Windows / iOS / Android（Unity）运行。
/// 也可通过构造函数传入 <see cref="INmpForwardEngine"/> 的加速实现（如 Unity Burst 版）。
/// 模型直接接收原始音频波形（CQT 已烘焙在模型图内）。
/// </summary>
public sealed class BasicPitchModel : IDisposable
{
    private readonly INmpForwardEngine _engine;

    public BasicPitchModel(INmpForwardEngine? engine = null)
    {
        _engine = engine ?? new NmpEngine();
    }

    /// <summary>
    /// 对整段音频进行推理，返回拼接后的 contour / note / onset 张量。
    /// </summary>
    /// <param name="audio">单声道 float 音频（采样率需为 22050 Hz）。</param>
    public ModelOutput Predict(float[] audio)
    {
        int totalSamples = audio.Length;
        var windows = EnumerateWindows(audio);

        var contours = new List<float[]>();
        var notes = new List<float[]>();
        var onsets = new List<float[]>();

        foreach (var window in windows)
        {
            var (c, n, o) = RunWindow(window);
            contours.Add(c);
            notes.Add(n);
            onsets.Add(o);
        }

        return new ModelOutput(
            Unwrap(contours, totalSamples),
            Unwrap(notes, totalSamples),
            Unwrap(onsets, totalSamples)
        );
    }

    /// <summary>
    /// 将音频切分为重叠窗口（每个 43844 样本，hop 36164）。
    /// </summary>
    private IEnumerable<float[]> EnumerateWindows(float[] audio)
    {
        int windowSize = Constants.AUDIO_N_SAMPLES;  // 43844
        int hop = Constants.HOP_SIZE;                // 36164
        int pad = Constants.OVERLAP_LEN / 2;         // 3840
        int cursor = -pad;

        while (cursor < audio.Length)
        {
            var window = new float[windowSize];
            int srcStart = Math.Max(0, cursor);
            int srcEnd = Math.Min(audio.Length, cursor + windowSize);
            int dstOffset = srcStart - cursor;
            int length = srcEnd - srcStart;
            if (length > 0)
                Array.Copy(audio, srcStart, window, dstOffset, length);
            // 窗口末尾不足部分已由数组默认零填充
            yield return window;
            cursor += hop;
        }
    }

    private (float[] contour, float[] note, float[] onset) RunWindow(float[] window)
    {
        _engine.Run(window);

        var contour = new float[_engine.BufferLength(NmpIr.BufferContour)];
        var note = new float[_engine.BufferLength(NmpIr.BufferNote)];
        var onset = new float[_engine.BufferLength(NmpIr.BufferOnset)];
        _engine.ReadBuffer(NmpIr.BufferContour, contour);
        _engine.ReadBuffer(NmpIr.BufferNote, note);
        _engine.ReadBuffer(NmpIr.BufferOnset, onset);
        return (contour, note, onset);
    }

    /// <summary>
    /// 拼接多窗口输出，去掉每端 15 帧的重叠边缘。
    /// 移植自 basic-pitch-dotnet ModelOutputHelper.Unwrap。
    /// </summary>
    private static Tensor Unwrap(List<float[]> outputs, int totalSamples)
    {
        if (outputs.Count == 0) return new Tensor(null, null);

        int nOlap = Constants.N_OVERLAPPING_FRAMES / 2;  // 15
        int nOutputFramesOri = totalSamples * Constants.ANNOTATIONS_FPS / Constants.AUDIO_SAMPLE_RATE;
        int step = outputs[0].Length / 172;  // 每帧的频率 bin 数（88 或 264）
        int framesPerWindow = 172;

        int rangeStart = nOlap * step;
        int rangeCount = (framesPerWindow - nOlap) * step - rangeStart;
        int framesPerWindowUsed = rangeCount / step;  // 142

        int totalFrames = Math.Min(outputs.Count * framesPerWindow - nOlap * 2, nOutputFramesOri);
        var data = new float[totalFrames * step];

        int size = 0;
        foreach (var win in outputs)
        {
            int copyLen = Math.Min(rangeCount, data.Length - size);
            if (copyLen <= 0) break;
            Array.Copy(win, rangeStart, data, size, copyLen);
            size += copyLen;
        }

        return new Tensor(data, new[] { totalFrames, step });
    }

    public void Dispose()
    {
        // 引擎不持有非托管资源，无需释放。
    }
}
