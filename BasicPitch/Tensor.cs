using System;

namespace BasicPitch;

/// <summary>
/// 简单的张量容器：float 数据 + 形状。
/// 替代 Windows.AI.MachineLearning 的 TensorFloat，保持平台无关。
/// </summary>
public sealed class Tensor
{
    public readonly float[]? Data;
    public readonly int[]? Shape;

    public Tensor(float[]? data, int[]? shape)
    {
        Data = data;
        Shape = shape;
    }

    public Tensor DeepClone()
    {
        float[]? data = null;
        int[]? shape = null;
        if (Data != null)
        {
            data = new float[Data.Length];
            Array.Copy(Data, data, Data.Length);
        }
        if (Shape != null)
        {
            shape = new int[Shape.Length];
            Array.Copy(Shape, shape, Shape.Length);
        }
        return new Tensor(data, shape);
    }
}

/// <summary>
/// 模型的三路输出：contour（弯音）、note（音符激活）、onset（起始点）。
/// </summary>
public sealed class ModelOutput
{
    public readonly Tensor Contours;
    public readonly Tensor Notes;
    public readonly Tensor Onsets;

    public ModelOutput(Tensor c, Tensor n, Tensor o)
    {
        Contours = c;
        Notes = n;
        Onsets = o;
    }
}
