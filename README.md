# MusicRecorder

音频转 MIDI / 五线谱的 C# 实现，包含转谱引擎与节奏检测模块，可按需引用。

- **BasicPitch** — 单乐器音高检测（Spotify Basic Pitch 移植）
- **Mt3** — 多乐器转谱（MR-MT3 模型，支持鼓组、钢琴、贝斯、吉他等）
- **Madmom** — 节奏 / 拍号检测（beat + downbeat + meter），madmom 模型的纯 C# IR 移植，详见 [`Madmom/README.md`](Madmom/README.md)

转谱引擎编译为**独立的 DLL**，互不依赖，可单独引用；Madmom 同为零原生依赖的 `netstandard2.1` 库。

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

Madmom/                 节奏/拍号检测库（beat+downbeat+meter）→ Madmom.dll（纯 C#，无原生依赖）
  MadmomAnalyzer.cs      高层入口（音频 → 特征 → 引擎 → DBN）
  MadmomEngine.cs        纯 C# 前向引擎（动态 T，支持 BLSTM/GRU/Dense）
  Features.cs            忠实移植 madmom 特征管线（多分辨率 STFT + 对数滤波组 + 正差分）
  Hmm.cs                 忠实移植 madmom ml/hmm.pyx（CSR 转移模型 + 对数域 Viterbi）
  BeatsHmm.cs            忠实移植 madmom features/beats_hmm.py（状态空间 + 转移/观测模型 + DBN processor）
  DownBeatDbn.cs         downbeat/meter DBN 封装（含拍号推断）
  BeatTracker.cs         beat-only DBN 封装
  IrGraph.cs             IR 图描述（算子编码 0..26 + 命名权重）
  *Ir.g.cs               由模型生成的权重文件（本机生成，不入库，见 Madmom/README.md）

Madmom.Test/            冒烟测试（合成权重，验证引擎 + DBN 管线，不依赖真实模型）
Madmom.EndToEnd/        端到端验证（真实权重 + 真实音频，含与 madmom 的对照校验）
.madmom_export/         madmom 模型 → IR → C# 权重 转换管线（export_madmom_ir.py / gen_madmom_cs.py）

mt3_onnx/               MT3 ONNX 模型正本（convert_mt3_onnx.py 的导出目录）
  mt3_encoder.onnx / mt3_decoder.onnx  模型文件

BasicPitchSharp.Test/    命令行测试程序（NAudio 读取音频）
Mp3ToSheet/              WPF 桌面应用 + 三种引擎的共用逻辑
  MainWindow.xaml(.cs)   界面：引擎下拉框，显示「解码 / 分析 / 合计」三段耗时与实时率
  AudioLoader.cs         NAudio 解码 → 22050Hz 单声道（无 WPF 依赖，与 Bench 共用）
  EngineRunner.cs        三引擎统一运行器 + 三段计时（无 WPF 依赖，与 Bench 共用）
Mp3ToSheet.Bench/        命令行测速器：解码一次，三引擎横向对比速度与结果
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
| `Madmom` | `Madmom\bin\<Config>\netstandard2.1\Madmom.dll` |
| `BasicPitchSharp.Test` | `BasicPitchSharp.Test\bin\<Config>\net8.0\BasicPitchSharp.Test.exe` |
| `Madmom.Test` | `Madmom.Test\bin\<Config>\net8.0\Madmom.Test.exe`（合成权重冒烟测试） |
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

### Madmom（节奏 / 拍号）

```csharp
using Madmom;

// 权重文件由本机转换脚本生成（纯个人/非商业用途，详见 Madmom/README.md 许可说明）
IrGraph downbeatModel = MadmomDownBeatIr.Build();   // 314 → 3
IrGraph beatModel = MadmomBeatIr.Build();           // 266 → 1（可选但推荐）
var analyzer = new MadmomAnalyzer(downbeatModel, beatModel);
var r = analyzer.Analyze(samples, sampleRate);

foreach (var b in r.Downbeats)
    Console.WriteLine($"{b.Time:F3}s  beatInBar={b.BeatInBar}  bpb={b.BeatsPerBar}  bpm={b.Bpm}");
```

模型文件生成（一次性、在你本机）：

```bash
pip install madmom
python .madmom_export/export_madmom_ir.py --model <downbeat.pkl> --fps 100 --out .madmom_export/out
python .madmom_export/gen_madmom_cs.py        # → Madmom/MadmomIr.g.cs
```

`Madmom.dll` 目标框架 `netstandard2.1`，可直接放入 Unity（含移动端），无 ONNX / 原生依赖。

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

### 三引擎横向测速（Mp3ToSheet.Bench）

命令行测速器。**只解码一次**，三引擎共用同一份 22050 Hz 音频，结果可直接横向比较：

```bat
dotnet run --project Mp3ToSheet.Bench -c Release -- <音频文件> [--engine all] [--chunk 10] [--out 目录]
```

| 参数 | 说明 |
|---|---|
| `--engine` | `all` 或逗号组合：`basicpitch,mt3,madmom`。默认 `all` |
| `--chunk` | 仅对 Madmom 生效，流式分块长度（秒）。`0` = 不分块（等价 madmom 一次性解码） |
| `--threshold` | BasicPitch 帧激活阈值，默认 `0.3` |
| `--out` | 输出目录，默认与音频同目录 |

WPF 版 `Mp3ToSheet` 与 Bench **共用** `AudioLoader.cs` / `EngineRunner.cs`，
所以 GUI 里看到的三段耗时与这里完全一致。

### 实测：333 秒流行歌曲（`凉凉`，13 MB MP3，48000 Hz → 22050 Hz）

Release / 桌面 32 核 / 每引擎单独跑（连跑会因 GC 堆膨胀与 CPU 争用互相干扰，故分开测）：

| 引擎 | 分析耗时 | 实时率 | 峰值内存 | 存活堆 | 产出 |
|---|---|---|---|---|---|
| BasicPitch | 126.5 s | 2.6× | 319 MB | 163 MB | 2855 个音符（单乐器） |
| MT3 | 74.9 s | 4.4× | 934 MB | 260 MB | 5224 个音符（多乐器，含鼓 1189） |
| **Madmom** | **21.7 s** | **14.7×** | 766 MB | 50 MB | 4/4 拍、BPM≈89.7、480 个拍点 |

### 为什么 BasicPitch（单乐器小模型）反而比 MT3（多乐器大模型）慢？

不是模型的锅，是**推理执行方式**的差距：

| | BasicPitch | MT3 |
|---|---|---|
| 推理内核 | `NmpEngine` —— **纯 C# 解释执行 IR，标量，无 SIMD** | **ONNX Runtime 原生 C++**（优化卷积 + 多线程） |
| 模型结构 | **CNN**，滑窗 43844 样本 / hop 36164（重叠 17.5%）→ 333 s 音频要跑 **~203 次前向** | 分段并行 + 自回归解码（KV-cache） |

Madmom 快则是因为模型本身很小（BLSTM 25 单元，约 9 万参数）且已做 SIMD 优化（`Simd.cs`）。
三者的速度差基本由「内核 + 模型结构」决定，**不代表转谱质量的高低**。

### Madmom 22 s 花在哪（`MADMOM_PROFILE=1` 分段计时，chunk=30）

| 阶段 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| **特征提取** `Features` | 12.3 s | **55%** | 三个分辨率（1024/2048/4096）× 100 fps 的 FFT + 对数滤波带 |
| BLSTM 前向 | 5.5 s | 24% | 时间维递增，只能并行门内点积 |
| DownBeat DBN 解码 | 3.8 s | 17% | Viterbi，托管实现 |
| Beat HMM | 0.8 s | 3% | — |

最大热点不是 RNN 而是特征提取，且逐帧独立 —— 这正是 Unity Burst 多核收益最大的部分（见下）。

### Unity Burst 双端内核（`Seams.cs` 接缝）

托管实现保持默认路径（桌面 Bench / 回归不变，已验证逐拍一致）；Unity 侧（gofire
`Standard Assets/Madmom/MadmomBurstEngine.cs`）通过 `Seams` 注入两个 Burst 内核：

| 接缝 | Unity 实现 | 多核策略 |
|---|---|---|
| `IFeatureEngine`（55% 热点） | `BurstFeatureEngine`：窗+FFT+幅度+滤波带+log 逐帧 `IJobParallelFor` | 按帧吃满所有核 |
| `Seams.Blstm`（IBlstmKernel） | `BurstBlstmKernel`：前向/后向两个 `IJob` 并行（写列区间不相交），门内点积 Burst 自动向量化 | 2 线程 |

数值语义对齐托管（门序 [i,f,c,o]、peephole、double 累加点积、log10 双精度）；
仅 FFT 蝶形用 float 存储，激活差异在低位、不影响拍点。数据（`ResolutionPlan` /
`BlockSpan` / `ResampleRange`）与托管 ExtractBlock 同源公开，Unity 侧不复制逻辑。

> ⚠️ 曾踩过的坑：早先测出 BasicPitch 328 s，是因为 `EngineRunner` 先调 `Convert()` 再调
> `ConvertToMidi()`，而后者**内部又跑了一次 `Convert()`**，完整推理算了两遍。
> 改为「`Convert()` 一次 + `new MidiWriter(notes).Write(...)`」后降到 126.5 s，产出音符不变。

- **解码**只占 1.0 s（330× 实时），重采样到 22050 Hz 不是瓶颈。
- 三者不是替代关系：**BasicPitch/MT3 出音符（音高），Madmom 出节拍与拍号**，
  精度与用途不同，速度也不该直接比。
- Madmom 三次重复测量为 21.7 / 21.7 / 22.2 s，很稳定。但**三引擎在同一进程内连跑会明显
  变慢**（Madmom 从 22 s 涨到 38 s）：前面的引擎把 GC 堆撑大、又持续占用 CPU。
  所以横向比较必须分开跑，别信单次连跑的数字。

Madmom 分块的精度代价见 `Madmom/README.md`「分块的精度代价」一节 —— 要点是
**块长别小于 30 秒**，否则拍号一致率会从 80% 掉到 65%。
