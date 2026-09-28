using System;
using System.Collections.Generic;

namespace BasicPitch;

/// <summary>
/// 后处理辅助函数，移植自 basic-pitch 的 note_creation.py。
/// </summary>
internal static class NotesHelper
{
    public static int HzToMidi(float freq)
    {
        return (int)Math.Round(12 * (Math.Log(freq, 2) - Math.Log(440.0, 2)) + 69);
    }

    public static float MidiToHz(int pitch)
    {
        return (float)(Math.Pow(2, (pitch - 69) / 12.0) * 440);
    }

    public static float ModelFrameToTime(int n)
    {
        if (n < 1) return 0f;
        float oriTime = (n * Constants.FFT_HOP) / (float)Constants.AUDIO_SAMPLE_RATE;
        float windowOffset = (float)Constants.FFT_HOP / (float)Constants.AUDIO_SAMPLE_RATE
            * ((float)Constants.ANNOT_N_FRAMES - (float)Constants.AUDIO_N_SAMPLES / (float)Constants.FFT_HOP)
            + 0.0018f;
        float v = (float)Math.Floor(n / (float)Constants.ANNOT_N_FRAMES) * windowOffset;
        return oriTime - v;
    }

    public static float MidiPitchToContourBin(int pitch)
    {
        float hz = MidiToHz(pitch);
        return 12f * Constants.CONTOURS_BINS_PER_SEMITONE
            * (float)Math.Log(hz / Constants.ANNOTATIONS_BASE_FREQUENCY, 2);
    }

    /// <summary>
    /// 根据频率范围过滤张量中的音高列。
    /// </summary>
    public static (Tensor, Tensor) ConstrainFrequency(Tensor onsets, Tensor frames, float? maxFreq, float? minFreq)
    {
        if (maxFreq == null && minFreq == null)
            return (onsets, frames);

        var newOnsets = onsets.DeepClone();
        var newFrames = frames.DeepClone();

        if (maxFreq != null)
        {
            int pitch = HzToMidi(maxFreq.Value) - Constants.MIDI_OFFSET;
            ZeroPitch(ref newOnsets, pitch, int.MaxValue);
            ZeroPitch(ref newFrames, pitch, int.MaxValue);
        }
        if (minFreq != null)
        {
            int pitch = HzToMidi(minFreq.Value) - Constants.MIDI_OFFSET;
            ZeroPitch(ref newOnsets, 0, pitch);
            ZeroPitch(ref newFrames, 0, pitch);
        }
        return (newOnsets, newFrames);
    }

    /// <summary>
    /// 将张量中 [startPitch, endPitch) 范围内的音高列置零。
    /// </summary>
    private static void ZeroPitch(ref Tensor tensor, int startPitch, int endPitch)
    {
        if (tensor.Data == null || tensor.Shape == null) return;
        var data = tensor.Data;
        int step = tensor.Shape[tensor.Shape.Length - 1];
        int nFrames = tensor.Shape[0];
        startPitch = Math.Max(0, startPitch);
        endPitch = Math.Min(step, endPitch);
        for (int f = 0; f < nFrames; f++)
        {
            int row = f * step;
            for (int p = startPitch; p < endPitch; p++)
                data[row + p] = 0f;
        }
    }

    /// <summary>
    /// 通过帧差异推断起始点（替代/增强模型的 onset 输出）。
    /// </summary>
    public static Tensor GetInferedOnsets(Tensor onsets, Tensor frames, int nDiff = 2)
    {
        if (frames.Data == null) return new Tensor(null, null);

        var frameData = frames.Data;
        int frameSize = (int)frames.Shape![frames.Shape.Length - 1];
        int totalFrameSize = frameData.Length;

        float[] diffs = new float[nDiff * totalFrameSize];
        var diffsSpan = diffs.AsSpan();
        for (int i = 0; i < nDiff; i++)
        {
            int start = i * totalFrameSize;
            int offset = frameSize * (i + 1);
            int length = Math.Max(totalFrameSize - offset, 0);
            if (length > 0)
                Array.Copy(frameData, 0, diffs, start + offset, length);
            var dest = diffsSpan.Slice(start, totalFrameSize);
            TensorOps.Subtract(frameData, dest, dest);
        }

        var frameDiff = diffsSpan.Slice(0, totalFrameSize);
        for (int i = 1; i < nDiff; i++)
            TensorOps.Min(diffsSpan.Slice(i * totalFrameSize, totalFrameSize), frameDiff, frameDiff);

        TensorOps.Max(frameDiff, 0f, frameDiff);
        diffsSpan.Slice(0, nDiff * frameSize).Clear();

        var onsetData = onsets.Data!;
        float maxDiff = TensorOps.Max(frameDiff);
        float scale = TensorOps.Max(onsetData);
        if (maxDiff != 0f) scale /= maxDiff;
        TensorOps.Multiply(frameDiff, scale, frameDiff);

        float[] ret = new float[onsetData.Length];
        TensorOps.Max(frameDiff, onsetData, ret);

        int[] shape = new int[onsets.Shape!.Length];
        Array.Copy(onsets.Shape, shape, shape.Length);
        return new Tensor(ret, shape);
    }

    /// <summary>
    /// 查找每列中超过阈值的局部极大值（对应 scipy.signal.argrelmax）。
    /// </summary>
    public static IList<int> FindValidOnsetIndexs(Tensor onsets, float threshold)
    {
        if (onsets.Shape![0] < 3) return new List<int>();

        var data = onsets.Data!;
        float[] mask = new float[data.Length];
        TensorOps.Min(data, threshold, mask);

        int step = (int)onsets.Shape[onsets.Shape.Length - 1];
        int limit = mask.Length - step;
        var ret = new List<int>();
        for (int i = step; i < limit; ++i)
        {
            if (mask[i] < threshold) continue;
            float v = data[i];
            if (v > data[i - step] && v > data[i + step])
                ret.Add(i);
        }
        return ret;
    }

    /// <summary>
    /// 生成高斯窗（scipy.signal.windows.gaussian）。
    /// </summary>
    public static float[] MakeGaussianWindow(int count, int std)
    {
        if (count <= 0) return Array.Empty<float>();
        if (count == 1) return new[] { 1.0f };

        var n = MathTool.ARange(-0.5f * (count - 1), 1.0f, count);
        float sig2 = std * std * 2f;
        TensorOps.Multiply(n, n, n);
        TensorOps.Divide(n, -sig2, n);
        TensorOps.Exp(n, n);
        return n;
    }
}
