using System;
using System.Collections.Generic;

namespace BasicPitch;

/// <summary>
/// 音频转 MIDI 的统一入口。
/// 在 Unity 中可直接传入 AudioClip 的 float[] 采样数据。
/// </summary>
public sealed class BasicPitchConverter : IDisposable
{
    private readonly BasicPitchModel _model;

    public BasicPitchConverter()
    {
        _model = new BasicPitchModel();
    }

    /// <summary>
    /// 使用自定义前向引擎（如 Unity Burst 加速版）构造。
    /// 除推理内核外，其余流程与默认构造函数完全一致。
    /// </summary>
    public BasicPitchConverter(INmpForwardEngine engine)
    {
        _model = new BasicPitchModel(engine);
    }

    /// <summary>
    /// 通过 <see cref="IAudioReader"/> 从文件读取音频并转换为音符列表。
    /// </summary>
    public List<Note> Convert(IAudioReader reader, string filePath, NotesConvertOptions? options = null)
    {
        var (samples, sampleRate) = reader.Read(filePath);
        return Convert(samples, sampleRate, options);
    }

    /// <summary>
    /// 通过 <see cref="IAudioReader"/> 读取文件并写入 MIDI。
    /// </summary>
    public void ConvertToMidi(
        IAudioReader reader, string filePath, string outputMidiPath,
        NotesConvertOptions? notesOptions = null, MidiWriteOptions? midiOptions = null)
    {
        var (samples, sampleRate) = reader.Read(filePath);
        ConvertToMidi(samples, sampleRate, outputMidiPath, notesOptions, midiOptions);
    }

    /// <summary>
    /// 将音频转换为音符列表。
    /// </summary>
    /// <param name="audio">音频采样数据（任意采样率、单声道）。</param>
    /// <param name="sampleRate">输入音频的采样率。</param>
    /// <param name="options">音符检测参数。</param>
    public List<Note> Convert(float[] audio, int sampleRate, NotesConvertOptions? options = null)
    {
        options ??= new NotesConvertOptions();

        // 重采样到 22050 Hz 单声道
        float[] mono = ToMono(audio);
        float[] resampled = Resample(mono, sampleRate, Constants.AUDIO_SAMPLE_RATE);

        // 模型推理
        var output = _model.Predict(resampled);

        // 后处理提取音符
        var converter = new NotesConverter(output);
        return converter.Convert(options.Value);
    }

    /// <summary>
    /// 将音频转换并写入 MIDI 文件。
    /// </summary>
    public void ConvertToMidi(
        float[] audio, int sampleRate, string outputMidiPath,
        NotesConvertOptions? notesOptions = null, MidiWriteOptions? midiOptions = null)
    {
        var notes = Convert(audio, sampleRate, notesOptions);
        midiOptions ??= new MidiWriteOptions();
        var writer = new MidiWriter(notes);
        writer.Write(outputMidiPath, midiOptions.Value);
    }

    /// <summary>
    /// 返回 MIDI 文件字节（Unity 中可直接写入 Application.persistentDataPath）。
    /// </summary>
    public byte[] ConvertToMidiBytes(
        float[] audio, int sampleRate,
        NotesConvertOptions? notesOptions = null, MidiWriteOptions? midiOptions = null)
    {
        var notes = Convert(audio, sampleRate, notesOptions);
        midiOptions ??= new MidiWriteOptions();
        var writer = new MidiWriter(notes);
        return writer.BuildBytes(midiOptions.Value);
    }

    /// <summary>
    /// 将多声道音频混合为单声道（取平均）。
    /// 若输入已是单声道则原样返回。
    /// </summary>
    /// <param name="channels">声道数（默认 1）。</param>
    public static float[] ToMono(float[] audio, int channels = 1)
    {
        if (channels <= 1) return audio;
        int n = audio.Length / channels;
        var mono = new float[n];
        for (int i = 0; i < n; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
                sum += audio[i * channels + c];
            mono[i] = sum / channels;
        }
        return mono;
    }

    /// <summary>
    /// 线性插值重采样。
    /// basic-pitch Python 版用 soxr_hq，此处用线性插值，结果基本一致。
    /// </summary>
    public static float[] Resample(float[] src, int srcRate, int dstRate)
    {
        if (srcRate == dstRate) return src;
        double ratio = (double)srcRate / dstRate;
        int dstLen = (int)Math.Ceiling(src.Length / ratio);
        var dst = new float[dstLen];
        for (int i = 0; i < dstLen; i++)
        {
            double srcIdx = i * ratio;
            int idx0 = (int)srcIdx;
            int idx1 = Math.Min(idx0 + 1, src.Length - 1);
            float frac = (float)(srcIdx - idx0);
            dst[i] = src[idx0] * (1 - frac) + src[idx1] * frac;
        }
        return dst;
    }

    public void Dispose() => _model.Dispose();
}
