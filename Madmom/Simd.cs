using System;
using System.Numerics;

namespace Madmom;

/// <summary>
/// 跨 Features / MadmomEngine 共用的 SIMD 辅助。
///
/// 移动端（ARM NEON）与本机的 AVX2 都是 128/256 位浮点 SIMD，用 System.Numerics.Vector
/// 写一次即可同时覆盖两端：Burst（Unity）会把同样的循环编译成 NEON，
/// 因此这里实测到的加速比可直接作为移动端可行性的参考。
/// </summary>
public static class Simd
{
    /// <summary>向量宽度（float 通道数）：AVX2 = 8，ARM NEON = 4。</summary>
    public static int Width => Vector<float>.Count;

    /// <summary>设环境变量 MADMOM_NOSIMD=1 可强制走标量路径，便于 A/B 对比。</summary>
    public static bool Enabled =>
        Vector.IsHardwareAccelerated &&
        Environment.GetEnvironmentVariable("MADMOM_NOSIMD") != "1";

    /// <summary>诊断信息，例如 "AVX2 8-wide"。</summary>
    public static string Describe =>
        !Vector.IsHardwareAccelerated ? "无硬件向量化"
        : Enabled ? $"向量化 {Vector<float>.Count}-wide"
        : "向量化已禁用(MADMOM_NOSIMD)";

    /// <summary>
    /// 点积 sum(a[aOff+i] * b[bOff+i])。向量化后求和顺序与标量不同，数值有 ~1e-7 相对误差。
    /// </summary>
    public static float Dot(float[] a, int aOff, float[] b, int bOff, int n)
    {
        float sum = 0f;
        int i = 0;
        if (Enabled && n >= Vector<float>.Count)
        {
            int w = Vector<float>.Count;
            var acc = Vector<float>.Zero;
            var va = new Span<float>(a);
            var vb = new Span<float>(b);
            for (; i <= n - w; i += w)
                acc += new Vector<float>(va.Slice(aOff + i, w)) * new Vector<float>(vb.Slice(bOff + i, w));
            for (int k = 0; k < w; k++) sum += acc[k];
        }
        for (; i < n; i++) sum += a[aOff + i] * b[bOff + i];
        return sum;
    }

    /// <summary>double 累加版点积：用于长向量（如滤波带 2048 项），保精度。</summary>
    public static double DotDouble(float[] a, int aOff, float[] b, int bOff, int n)
    {
        double sum = 0;
        int i = 0;
        if (Enabled && n >= Vector<float>.Count)
        {
            int w = Vector<float>.Count;
            var acc = Vector<float>.Zero;
            var va = new Span<float>(a);
            var vb = new Span<float>(b);
            for (; i <= n - w; i += w)
                acc += new Vector<float>(va.Slice(aOff + i, w)) * new Vector<float>(vb.Slice(bOff + i, w));
            for (int k = 0; k < w; k++) sum += acc[k];
        }
        for (; i < n; i++) sum += (double)a[aOff + i] * b[bOff + i];
        return sum;
    }
}
