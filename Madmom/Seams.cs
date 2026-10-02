using System;

namespace Madmom;

/// <summary>
/// 双端内核接缝：默认全托管实现（桌面 .NET 跑 Bench/回归）；Unity 侧把 Burst 实现
/// 注入这两个口子即可，托管代码零改动（见 gofire Standard Assets/Madmom/MadmomBurstEngine.cs）。
/// </summary>
public static class Seams
{
    /// <summary>BLSTM 整层内核（Unity Burst 实现，双向两线程并行 + SIMD 点积）。null = 托管实现。</summary>
    public static IBlstmKernel? Blstm;

    /// <summary>构造特征引擎（Unity 返回 Burst 逐帧多核实现；桌面返回 null 走托管）。</summary>
    public static System.Func<IFeatureEngine>? FeatureEngineFactory;
}

/// <summary>特征提取引擎接缝：语义与 <see cref="Features.ExtractBlock"/> 完全一致。</summary>
public interface IFeatureEngine
{
    float[][] ExtractBlock(float[] audio, int sampleRate, Features.FeatureConfig cfg,
        int frameStart, int frameCount, out int featureDim);
}

    /// <summary>
    /// BLSTM 整层内核接缝。x 为 (T,F) 输入（自 xOff 起连续）；dst 为 (T,2H) 输出，
    /// 前向占列 [0,H)、后向占列 [H,2H)。hFwd/cFwd 为前向 LSTM/GRU 状态：
    /// 调用前为初值（0 或流式上一块末值），返回时须写入末值。pe 为 LSTM peephole 权重（可空）。
    /// </summary>
    public interface IBlstmKernel
    {
        void Run(float[] x, int xOff, int T, int F,
            float[] Wxf, float[] Whf, float[] bf,
            float[] Wxb, float[] Whb, float[] bb,
            float[] pe, int H, Span<float> dst, int outF,
            float[] hFwd, float[] cFwd);
    }
