using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// 常量折叠后的前向图描述（与 BasicPitch 的 NmpIr 同构，但额外支持 RNN 算子）。
///
/// 数据来源二选一：
///  - 由 <c>gen_madmom_cs.py</c> 生成的 <c>MadmomIr.g.cs</c> 静态常量（零依赖，Unity 友好，推荐）；
///  - 运行时由转换器产物（ir.json + weights.bin）解析后构造（调试/换模型用，需 System.Text.Json，放在测试工程）。
///
/// 缓冲区策略：所有张量（含中间激活 + 权重常量）统一由引擎在一段连续 float 数组上分配。
/// </summary>
public sealed class IrGraph
{
    // ---- 算子编码（必须与 Python 侧 gen_madmom_cs.py / export_madmom_ir.py 完全一致）----
    public const byte OpReshape = 0;
    public const byte OpSlice = 1;
    public const byte OpPad = 2;
    public const byte OpUnsqueeze = 3;
    public const byte OpConv = 4;
    public const byte OpNeg = 5;
    public const byte OpTranspose = 6;
    public const byte OpConcat = 7;
    public const byte OpMul = 8;
    public const byte OpReduceSum = 9;
    public const byte OpSqrt = 10;
    public const byte OpAdd = 11;
    public const byte OpLog = 12;
    public const byte OpReduceMin = 13;
    public const byte OpSub = 14;
    public const byte OpReduceMax = 15;
    public const byte OpDiv = 16;
    public const byte OpEqual = 17;
    public const byte OpWhere = 18;
    public const byte OpRelu = 19;
    public const byte OpSigmoid = 20;
    public const byte OpLstm = 21;
    public const byte OpGru = 22;
    public const byte OpBlstm = 23;
    public const byte OpDense = 24;
    public const byte OpTanh = 25;
    public const byte OpSoftmax = 26;

    // ---- 图拓扑 ----
    public readonly int NBuffers;
    public readonly int NNodes;
    public readonly int NWeights;
    public readonly int InputBuffer;
    /// <summary>模型输入特征维度（第一层权重决定）。由生成常量时填充。</summary>
    public int InputFeatureDim;
    /// <summary>模型输出维度（downbeat=3：beat/downbeat/position；beat=1）。由生成常量时填充。</summary>
    public int OutputDim;
    /// <summary>特征帧率（downbeat 默认 100）。由生成常量时填充。</summary>
    public double Fps;
    /// <summary>模型输出所在的缓冲区索引。由生成常量时填充。</summary>
    public int OutputBuffer;
    public readonly int[] BufferCount;
    public readonly int[] BufferRank;
    public readonly int[] ShapeOff;
    public readonly int[] ShapeData;

    public readonly byte[] Op;
    public readonly int[] In0, In1, In2, Out;
    public readonly int[] ListOff;
    public readonly int[] List;

    public readonly int[] WeightOff;
    public readonly int[] WeightCount;
    public readonly int[] WeightRank;
    public readonly int[] WeightShapeOff;
    public readonly int[] WeightShapeData;
    public readonly float[] WeightData;

    /// <summary>RNN 节点按名字引用权重（如 BLSTM 节点 3 的 "lstm_3fx"/"lstm_3fh"/"lstm_3fb"…）；Dense 节点 n 用 "dense{n}_x"/"dense{n}_b"。由生成常量时填充，命名须与 MadmomEngine 完全一致。</summary>
    public readonly Dictionary<string, float[]> Weights;

    public IrGraph(
        int nBuffers, int nNodes, int nWeights, int inputBuffer,
        int[] bufferCount, int[] bufferRank, int[] shapeOff, int[] shapeData,
        byte[] op, int[] in0, int[] in1, int[] in2, int[] @out,
        int[] listOff, int[] list,
        int[] weightOff, int[] weightCount, int[] weightRank,
        int[] weightShapeOff, int[] weightShapeData, float[] weightData,
        Dictionary<string, float[]>? weights = null)
    {
        NBuffers = nBuffers;
        NNodes = nNodes;
        NWeights = nWeights;
        InputBuffer = inputBuffer;
        BufferCount = bufferCount;
        BufferRank = bufferRank;
        ShapeOff = shapeOff;
        ShapeData = shapeData;
        Op = op;
        In0 = in0;
        In1 = in1;
        In2 = in2;
        Out = @out;
        ListOff = listOff;
        List = list;
        WeightOff = weightOff;
        WeightCount = weightCount;
        WeightRank = weightRank;
        WeightShapeOff = weightShapeOff;
        WeightShapeData = weightShapeData;
        WeightData = weightData;
        Weights = weights ?? new Dictionary<string, float[]>();
    }

    /// <summary>按名字取权重张量（RNN/Dense 用）。</summary>
    public float[] W(string name)
        => Weights.TryGetValue(name, out var w)
            ? w
            : throw new KeyNotFoundException("权重不存在: " + name);

    public static int Prod(int[] shape)
    {
        int p = 1;
        foreach (var x in shape) p *= x;
        return p;
    }
}
