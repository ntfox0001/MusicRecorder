using System;
using System.IO;
using System.Text;

namespace Madmom.EndToEnd;

/// <summary>
/// 极简 WAV（RIFF）解码器：把 PCM / IEEE-float WAV 解成单声道 float32 [-1,1] 与采样率。
/// 仅用于本地端到端冒烟测试，不追求覆盖全部 WAV 变体。
/// </summary>
public static class WavLoader
{
    public static (float[] Audio, int SampleRate) Load(string path)
    {
        byte[] buf = File.ReadAllBytes(path);
        if (buf.Length < 44) throw new InvalidDataException("文件过短，不是有效 WAV");
        // 注意：GetString(bytes, byteIndex, byteCount)，第三参数是“字节数”不是“结束下标”
        if (Encoding.ASCII.GetString(buf, 0, 4) != "RIFF" ||
            Encoding.ASCII.GetString(buf, 8, 4) != "WAVE")
            throw new InvalidDataException("不是 RIFF/WAVE 文件");

        int pos = 12;
        int audioFormat = 1, channels = 1, sampleRate = 44100, bitsPerSample = 16;
        int dataOffset = -1, dataSize = 0;

        while (pos + 8 <= buf.Length)
        {
            string id = Encoding.ASCII.GetString(buf, pos, 4);
            int size = BitConverter.ToInt32(buf, pos + 4);
            int body = pos + 8;
            if (body > buf.Length) break;

            if (id == "fmt ")
            {
                audioFormat = BitConverter.ToUInt16(buf, body + 0);
                channels = BitConverter.ToUInt16(buf, body + 2);
                sampleRate = BitConverter.ToInt32(buf, body + 4);
                bitsPerSample = BitConverter.ToUInt16(buf, body + 14);
            }
            else if (id == "data")
            {
                dataOffset = body;
                dataSize = Math.Min(size, buf.Length - body);
                if (dataSize <= 0) dataSize = buf.Length - body;
            }

            pos = body + size + (size % 2); // chunk 按偶数字节对齐
        }

        if (dataOffset < 0) throw new InvalidDataException("未找到 data chunk");
        if (channels < 1) channels = 1;
        int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample < 1) bytesPerSample = 1;
        int frameCount = dataSize / (bytesPerSample * channels);

        var mono = new float[frameCount];
        int p = dataOffset;
        for (int f = 0; f < frameCount; f++)
        {
            double sum = 0;
            for (int c = 0; c < channels; c++)
            {
                double v = ReadSample(buf, p, audioFormat, bitsPerSample);
                p += bytesPerSample;
                sum += v;
            }
            mono[f] = (float)(sum / channels);
        }
        return (mono, sampleRate);
    }

    private static double ReadSample(byte[] buf, int off, int audioFormat, int bitsPerSample)
    {
        // IEEE float（format=3）
        if (audioFormat == 3)
        {
            if (bitsPerSample == 64) return BitConverter.ToDouble(buf, off);
            return BitConverter.ToSingle(buf, off);
        }
        // PCM 整数
        switch (bitsPerSample)
        {
            case 8:
                return (buf[off] - 128) / 128.0;          // 8-bit 为无符号偏移
            case 16:
                return BitConverter.ToInt16(buf, off) / 32768.0;
            case 24:
                {
                    int v = buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16);
                    if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000); // 符号扩展
                    return v / 8388608.0;
                }
            case 32:
                return BitConverter.ToInt32(buf, off) / 2147483648.0;
            default:
                throw new NotSupportedException($"不支持的位深: {bitsPerSample}");
        }
    }
}
