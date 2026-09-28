using System;
using System.IO;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

/// <summary>
/// 基于 NAudio 的音频读取器，同时实现 BasicPitch 和 Mt3 的 IAudioReader 接口。
/// 读取任意格式音频文件，返回单声道 float 采样数据和原始采样率。
/// </summary>
public sealed class NAudioAudioReader : BasicPitch.IAudioReader, Mt3.IAudioReader
{
    public (float[] Samples, int SampleRate) Read(string filePath)
    {
        using var reader = new AudioFileReader(filePath);
        int channels = reader.WaveFormat.Channels;
        int sampleRate = reader.WaveFormat.SampleRate;

        using var ms = new MemoryStream();
        var buf = new float[sampleRate * channels * 2];
        int read;
        while ((read = reader.Read(buf, 0, buf.Length)) > 0)
            for (int i = 0; i < read; i++)
                ms.Write(BitConverter.GetBytes(buf[i]), 0, 4);

        var bytes = ms.ToArray();
        var interleaved = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, interleaved, 0, bytes.Length);

        if (channels == 1)
            return (interleaved, sampleRate);

        int n = interleaved.Length / channels;
        var mono = new float[n];
        for (int i = 0; i < n; i++)
        {
            float s = 0;
            for (int c = 0; c < channels; c++) s += interleaved[i * channels + c];
            mono[i] = s / channels;
        }
        return (mono, sampleRate);
    }
}
