using System.Collections.Generic;
using System.IO;
using BasicPitch;
using BasicPitch.Burst;
using UnityEngine;

/// <summary>
/// Unity 示例：将 AudioClip 转换为 MIDI / 音符列表。
///
/// 前置准备：
///   1. 将 BasicPitch.dll 放入 Assets/Plugins/，并勾选目标平台（Windows / iOS / Android）；
///   2. Package Manager 安装 Burst（com.unity.burst）。
///
/// 说明：模型参数已固化在 DLL 内，无需任何 ONNX 模型文件或原生推理库。
/// 推理走 Burst + Job System（<see cref="NmpBurstEngine"/>），移动端可多核并行。
/// </summary>
public class AudioToMidi : MonoBehaviour
{
    [Header("输入音频")]
    public AudioClip audioClip;

    [Header("检测参数")]
    [Range(0.05f, 0.95f)] public float frameThreshold = 0.3f;
    [Range(0.05f, 0.95f)] public float onsetThreshold = 0.5f;

    [Header("输出")]
    public string midiFileName = "output.mid";

    private NmpBurstEngine _engine;
    private BasicPitchConverter _converter;

    void Start()
    {
        // Burst 引擎持有约 33MB 的 NativeArray，创建一次、跨多次转换复用
        _engine = new NmpBurstEngine();
        _converter = new BasicPitchConverter(_engine);
        Debug.Log("[AudioToMidi] 初始化完成（Burst 引擎）");
    }

    /// <summary>
    /// 在 Inspector 中点击右键菜单或调用此方法执行转换。
    /// </summary>
    [ContextMenu("Convert to MIDI")]
    public void ConvertToMidi()
    {
        if (audioClip == null)
        {
            Debug.LogError("[AudioToMidi] 请先指定 AudioClip");
            return;
        }

        // 从 AudioClip 获取 float 采样（交错多声道）
        float[] samples = new float[audioClip.samples * audioClip.channels];
        audioClip.GetData(samples, 0);

        // 混合为单声道
        float[] mono = BasicPitchConverter.ToMono(samples, audioClip.channels);

        var options = new NotesConvertOptions
        {
            FrameThreshold = frameThreshold,
            OnsetThreshold = onsetThreshold,
        };

        // 转换为音符列表
        List<Note> notes = _converter.Convert(mono, audioClip.frequency, options);
        Debug.Log($"[AudioToMidi] 检测到 {notes.Count} 个音符");
        foreach (var n in notes)
            Debug.Log($"  MIDI {n.Pitch}: {n.StartTime:F2}s - {n.EndTime:F2}s");

        // 保存 MIDI 到持久化目录
        string midiPath = Path.Combine(Application.persistentDataPath, midiFileName);
        _converter.ConvertToMidi(mono, audioClip.frequency, midiPath, options);
        Debug.Log($"[AudioToMidi] MIDI 已保存: {midiPath}");
    }

    void OnDestroy()
    {
        _converter?.Dispose();
        _engine?.Dispose();
    }
}
