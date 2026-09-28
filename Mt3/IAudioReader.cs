namespace Mt3;

/// <summary>
/// 音频读取接口。将文件读取逻辑与转谱核心解耦，
/// 调用方可以基于 NAudio、Unity AudioClip 或其他音频库实现。
/// </summary>
public interface IAudioReader
{
    /// <summary>
    /// 从文件读取音频，返回单声道 float 采样数据和原始采样率。
    /// </summary>
    /// <param name="filePath">音频文件路径。</param>
    /// <returns>(采样数据, 采样率)。</returns>
    (float[] Samples, int SampleRate) Read(string filePath);
}
