using System;
using System.IO;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Mp3ToSheet;

/// <summary>
/// 音频解码 → 22050 Hz 单声道 float[]。不依赖 WPF，供 Mp3ToSheet (GUI) 与 Bench (CLI) 共用。
/// </summary>
public static class AudioLoader
{
    public const int TargetRate = 22050;

    /// <summary>用 NAudio 解码（mp3/wav/flac 通吃）+ WDL 重采样到 22050 Hz，混成单声道。</summary>
    public static (float[] audio, int srcRate) Load(string input, int targetRate = TargetRate)
    {
        using var reader = new AudioFileReader(input);
        int srcRate = reader.WaveFormat.SampleRate;
        // 注意：WdlResamplingSampleProvider 不实现 IDisposable，不能写 using
        var resampler = new WdlResamplingSampleProvider(reader, targetRate);

        int channels = resampler.WaveFormat.Channels;
        int sr = resampler.WaveFormat.SampleRate;

        // 估算输出长度：源文件字节数 → 采样数 → 目标采样数 → 乘声道
        int bytesPerSample = Math.Max(1, reader.WaveFormat.BitsPerSample / 8);
        long srcSamples = reader.Length / (long)bytesPerSample / Math.Max(1, reader.WaveFormat.Channels);
        long estimated = srcSamples * (long)targetRate / Math.Max(1, srcRate) * channels;

        var buf = new float[(int)Math.Clamp(estimated + channels, 4096, int.MaxValue)];
        int total = 0;
        var chunk = new float[sr * channels]; // 1 秒
        int read;
        while ((read = resampler.Read(chunk, 0, chunk.Length)) > 0)
        {
            if (total + read > buf.Length)
                Array.Resize(ref buf, (int)Math.Min((long)buf.Length * 2 + read, int.MaxValue));
            Array.Copy(chunk, 0, buf, total, read);
            total += read;
        }

        float[] audio;
        if (channels == 1)
        {
            audio = new float[total];
            Array.Copy(buf, audio, total);
        }
        else
        {
            int n = total / channels;
            audio = new float[n];
            for (int i = 0; i < n; i++)
            {
                float sum = 0;
                int b = i * channels;
                for (int c = 0; c < channels; c++) sum += buf[b + c];
                audio[i] = sum / channels;
            }
        }
        return (audio, srcRate);
    }
}
