using System;

namespace BasicPitch;

/// <summary>
/// 为 netstandard2.1 补充缺失的向量原语（TensorPrimitives 的替代），
/// 用简单循环实现，保证 Unity / 移动端兼容。
/// </summary>
internal static class TensorOps
{
    /// <summary>逐元素相乘：dest[i] = x[i] * y[i]。</summary>
    public static void Multiply(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = x[i] * y[i];
    }

    /// <summary>返回最大值的索引；长度为 0 时返回 0。</summary>
    public static int IndexOfMax(ReadOnlySpan<float> x)
    {
        if (x.Length == 0) return 0;
        int best = 0;
        float bv = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > bv) { bv = x[i]; best = i; }
        }
        return best;
    }

    public static void Max(ReadOnlySpan<float> x, float y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Max(x[i], y);
    }

    /// <summary>逐元素取较大者：dest[i] = max(x[i], y[i])。</summary>
    public static void Max(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Max(x[i], y[i]);
    }

    /// <summary>返回整个 span 的最大值；为空时返回 <see cref="float.NegativeInfinity"/>。</summary>
    public static float Max(ReadOnlySpan<float> x)
    {
        if (x.Length == 0) return float.NegativeInfinity;
        float m = x[0];
        for (int i = 1; i < x.Length; i++)
            if (x[i] > m) m = x[i];
        return m;
    }

    /// <summary>逐元素取较小者：dest[i] = min(x[i], y[i])。</summary>
    public static void Min(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Min(x[i], y[i]);
    }

    /// <summary>逐元素相减：dest[i] = x[i] - y[i]。</summary>
    public static void Subtract(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = x[i] - y[i];
    }

    /// <summary>逐元素指数：dest[i] = exp(x[i])。</summary>
    public static void Exp(ReadOnlySpan<float> x, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Exp(x[i]);
    }

    public static void Min(ReadOnlySpan<float> x, float y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Min(x[i], y);
    }

    public static void Multiply(ReadOnlySpan<float> x, float y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = x[i] * y;
    }

    public static void Add(ReadOnlySpan<float> x, float y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = x[i] + y;
    }

    public static void Divide(ReadOnlySpan<float> x, float y, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = x[i] / y;
    }

    public static void Round(ReadOnlySpan<float> x, Span<float> dest)
    {
        for (int i = 0; i < x.Length; i++)
            dest[i] = MathF.Round(x[i]);
    }
}
