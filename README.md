# MusicRecorder

音频转 MIDI / 五线谱的 C# 实现，包含两套独立的转谱引擎，可按需引用。

- **BasicPitch** — 单乐器音高检测（Spotify Basic Pitch 移植）
- **Mt3** — 多乐器转谱（MR-MT3 模型，支持鼓组、钢琴、贝斯、吉他等）

两套引擎编译为**独立的 DLL**，互不依赖，可单独引用。

## 项目结构

```
BasicPitch/              单乐器转谱库 → BasicPitch.dll
  BasicPitchConverter.cs  入口类
  BasicPitchModel.cs      推理调度（窗口切分 / 重叠拼接）
  NmpEngine.cs            纯 C# 前向引擎 + INmpForwardEngine 接口（无 ONNX Runtime 依赖）
  NmpIr.g.cs              由 nmp.onnx 生成的前向图描述 + 权重常量
  NotesConverter.cs       音符后处理
  Note.cs / MidiWriter.cs 音符与 MIDI 输出
  IAudioReader.cs         音频读取接口

Mt3/                     多乐器转谱库 → Mt3.dll
  Mt3Converter.cs         入口类（encoder + decoder）
  Mt3Codec.cs             token 编解码 + Mt3Note
  MelSpectrogram.cs       log-mel 频谱
  Mt3MidiWriter.cs        多乐器 MIDI 输出
  IAudioReader.cs         音频读取接口
  mel_filterbank.bin / mel_window.bin   mel 滤波器（作为嵌入资源编入 Mt3.dll）

mt3_onnx/               MT3 ONNX 模型正本（convert_mt3_onnx.py 的导出目录）
  mt3_encoder.onnx / mt3_decoder.onnx  模型文件

BasicPitchSharp.Test/    命令行测试程序（NAudio 读取音频）
Mp3ToSheet/              WPF 桌面应用（BasicPitch）
UnityExample/            Unity 集成示例
  Assets/Scripts/BasicPitchBurst/  Burst 加速引擎（NmpBurstEngine，需 Burst 包）

build.bat                一键构建全部工程
convert_mt3_onnx.py      MT3 PyTorch checkpoint → ONNX 导出
export_mt3_onnx.bat      上述导出脚本的一键包装（自动准备虚拟环境）
.onnx_export/            BasicPitch 前向图生成管线（nmp.onnx → NmpIr.g.cs）
```

## 构建

```bat
build.bat            :: Release，构建 BasicPitch / Mt3 / 测试程序 / WPF 应用
build.bat Debug      :: Debug
```

产物：

| 工程 | 输出 |
|------|------|
| `BasicPitch` | `BasicPitch\bin\<Config>\netstandard2.1\BasicPitch.dll` |
| `Mt3` | `Mt3\bin\<Config>\netstandard2.1\Mt3.dll` |
| `BasicPitchSharp.Test` | `BasicPitchSharp.Test\bin\<Config>\net8.0\BasicPitchSharp.Test.exe` |
| `Mp3ToSheet` | `Mp3ToSheet\bin\<Config>\net8.0-windows\Mp3ToSheet.exe` |

## 快速开始

### 引用

```xml
<!-- 单乐器 -->
<ProjectReference Include="..\BasicPitch\BasicPitch.csproj" />

<!-- 多乐器 -->
<ProjectReference Include="..\Mt3\Mt3.csproj" />
```

### BasicPitch（单乐器）

```csharp
using BasicPitch;

// 模型已内置，无需任何模型文件
using var bp = new BasicPitchConverter();

// 传入 PCM 采样 + 采样率
List<Note> notes = bp.Convert(samples, sampleRate);

// 或直接输出 MIDI
bp.ConvertToMidi(samples, sampleRate, "output.mid");
```

### Mt3（多乐器）

```csharp
using Mt3;

// 从文件路径加载模型（桌面端）
using var mt3 = new Mt3Converter("mt3_encoder.onnx", "mt3_decoder.onnx");

// 或从 byte[] 加载（Unity / 移动端推荐）
using var mt3 = new Mt3Converter(encoderBytes, decoderBytes);

List<Mt3Note> notes = mt3.Convert(samples, sampleRate);
mt3.ConvertToMidi(samples, sampleRate, "output.mid");
```

## 音频读取（IAudioReader）

两个库都定义了 `IAudioReader` 接口，将音频解码与核心转谱解耦：

```csharp
public interface IAudioReader
{
    (float[] Samples, int SampleRate) Read(string filePath);
}
```

调用方负责实现音频解码（NAudio、Unity AudioClip 等）。Test 项目提供了基于 NAudio 的 `NAudioAudioReader` 实现。

使用 reader 重载：

```csharp
var reader = new NAudioAudioReader();   // 或自定义实现
var notes = bp.Convert(reader, "song.mp3");
mt3.ConvertToMidi(reader, "song.mp3", "out.mid");
```

## Unity 集成

两套引擎目标框架均为 `netstandard2.1`，可直接放入 Unity 使用，但依赖不同：

| 引擎 | 依赖 | 适用平台 |
|------|------|----------|
| BasicPitch | 无原生依赖（模型参数已固化在 DLL 内） | Windows / iOS / Android |
| Mt3 | `Microsoft.ML.OnnxRuntime` + 各平台原生库 | 桌面端（移动端算力不足） |

### BasicPitch（免原生依赖）

只需 `BasicPitch.dll`，无需任何 ONNX 模型或推理库：

```csharp
using BasicPitch;

// 音频可来自 AudioClip / UnityWebRequest，解码后得到 float[] + 采样率
var converter = new BasicPitchConverter();
var notes = converter.Convert(samples, sampleRate);
```

#### Burst 加速（移动端）

桌面默认的纯 C# 引擎单窗口约 600 ms，手机上偏慢，因此提供了 Burst + Job System 版本：

1. Package Manager 安装 Burst（`com.unity.burst`）；
2. 把 `UnityExample/Assets/Scripts/BasicPitchBurst/` 整个文件夹放入你的工程；
3. 用 `NmpBurstEngine` 构造转换器：

```csharp
using BasicPitch;
using BasicPitch.Burst;

using var engine = new NmpBurstEngine();     // 持有约 33MB NativeArray，创建一次即可复用
using var converter = new BasicPitchConverter(engine);
var notes = converter.Convert(samples, sampleRate);
```

Burst 版与桌面版数值等价，仅执行方式不同：逐元素 / 布局 / 规约算子按输出元素并行，卷积按输出元素多维并行。

### Mt3（需 ONNX Runtime）

Mt3 依赖 `Microsoft.ML.OnnxRuntime`，且模型文件需随工程分发。

#### 模型加载（兼容 Android StreamingAssets）

Unity 中 `StreamingAssets` 在 Android 上位于 APK 内部，无法用 `File.ReadAllBytes` 读取。需通过 `UnityWebRequest` 读取为 `byte[]` 后传入：

```csharp
IEnumerator LoadAndRun()
{
    // 读取 ONNX 模型字节
    var encReq = UnityWebRequest.Get(Path.Combine(Application.streamingAssetsPath, "mt3_encoder.onnx"));
    yield return encReq.SendWebRequest();
    byte[] encBytes = encReq.downloadHandler.data;

    var decReq = UnityWebRequest.Get(Path.Combine(Application.streamingAssetsPath, "mt3_decoder.onnx"));
    yield return decReq.SendWebRequest();
    byte[] decBytes = decReq.downloadHandler.data;

    using var mt3 = new Mt3Converter(encBytes, decBytes);

    // 读取音频（MP3 用 AudioClip 解码）
    var audioReq = UnityWebRequestMultimedia.GetAudioClip(
        Path.Combine(Application.streamingAssetsPath, "song.mp3"), AudioType.MPEG);
    yield return audioReq.SendWebRequest();
    var clip = DownloadHandlerAudioClip.GetContent(audioReq);
    var samples = new float[clip.samples];
    clip.GetData(samples, 0);

    var notes = mt3.Convert(samples, clip.frequency);
}
```

### 依赖部署

**BasicPitch**：只需将 `BasicPitch.dll` 放入 `Assets/Plugins/`。

**Mt3**：需将以下文件放入 `Assets/Plugins/`：
- `Mt3.dll`
- `Microsoft.ML.OnnxRuntime.dll`
- 对应平台的 onnxruntime 原生库（`onnxruntime.dll` / `.dylib` / `.so`）

Mt3 的 ONNX 模型文件放入 `Assets/StreamingAssets/`。

## MT3 模型

MT3 需要额外的 ONNX 模型文件（约 350 MB），体积过大，**未随本仓库分发**。
BasicPitch 的模型（`BasicPitch/nmp.onnx`，230 KB）参数已固化进 `NmpIr.g.cs`，无需额外下载。

### 1. 下载 MR-MT3 checkpoint

模型来自 HuggingFace 仓库 [`gudgud1014/MR-MT3`](https://huggingface.co/gudgud1014/MR-MT3)（MIT 许可）：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download gudgud1014/MR-MT3 --local-dir mr_mt3_model
```

或用 `mt3-infer` 自动下载：

```bash
pip install mt3-infer
mt3-infer download mr_mt3
```

本工程使用其中的 `mt3.pth`，请确保路径为 `mr_mt3_model/mt3.pth`。

### 2. 导出为 ONNX

```bat
export_mt3_onnx.bat
```

脚本首次运行会自动创建 `.onnx_conv_venv` 虚拟环境并安装
`torch / transformers==4.44.0 / onnx / onnxruntime / mt3-infer`，然后调用
`convert_mt3_onnx.py`，产物写入 `mt3_onnx/`：

- `mt3_encoder.onnx`（+ `.onnx.data`）
- `mt3_decoder.onnx`（+ `.onnx.data`，带 KV-cache 优化）

也可以在已配置好的 Python 环境中直接运行 `python convert_mt3_onnx.py`。

之后 `BasicPitchSharp.Test` 构建时会自动把 `mt3_onnx/` 下的模型复制到输出目录。

## 性能

MT3 转谱采用分段并行推理 + 自回归解码（KV-cache），4 分钟音频约 135 秒（CPU）。
