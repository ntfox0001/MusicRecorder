# Madmom（C# 节奏 / 拍号检测）

`madmom` 风格的 **beat（拍点）+ downbeat（小节首拍）+ meter（拍号/4-4、3-4…）** 检测，纯 C# 实现，
采用与本项目 `BasicPitch` **完全相同的「常量折叠 IR + 权重内联」方案**：不依赖 ONNX Runtime、不依赖
任何原生库，模型权重由你本地从 madmom 导出后固化进一个 `.g.cs` 文件，桌面 / Unity / 移动端通用。

```
audio(float[], sampleRate)
   └─ Features.Extract        多分辨率 STFT + 对数滤波组 + log + 正差分（忠实移植 madmom 特征管线）
        └─ MadmomEngine.Run   BLSTM/Dense/Softmax 前向（纯 C# 解释执行 IR）
             └─ Hmm.cs + BeatsHmm.cs   DBN/HMM：Bar/Beat 状态空间 + 指数 tempo 转移 + 对数域 Viterbi
                  └─ List<Beat>{ Time, BeatInBar, BeatsPerBar, Bpm }
```

---

## 许可（事实说明，非阻断项）

本项目为**纯个人 / 非商业**用途。

- 上游 madmom 及其训练权重以 **CC BY-NC-SA 4.0（非商业）** 发布。本仓库**已内置**从 madmom 0.16.1
  默认模型导出的 `MadmomDownBeatIr.g.cs` / `MadmomBeatIr.g.cs`（内联权重），clone 后即可编译运行，
  使用时请遵守上游的非商业条款；也可按下方步骤用你自己的 madmom 重新生成覆盖。
- 引擎、特征、DBN/HMM 等**本仓库自研代码**为 MIT（与项目其余部分一致），可自由使用；
  受上游协议约束的**仅**是权重数据本身。若将来要商用，联系原作者取授权或自行训练等价模型替换即可。

---

## 重新生成模型（可选）

权重已内置，以下步骤仅在你想用**自己的** madmom 重新生成时执行。需要 Python 3.8+ 与本地
`madmom`（`pip install madmom`，它本身依赖 numpy/scipy/cython）。

### 1. 取 madmom 的模型 `.pkl`

madmom 通过 `madmom.ml.nn.NeuralNetwork.load(path)` 载入。模型文件随 madmom 包分发，路径因版本而异，
用下面命令定位：

```bash
python -c "import madmom, os; print(os.path.join(os.path.dirname(madmom.__file__),'models'))"
```

常用文件（名称随版本变化，以你本机为准）：

本机 madmom **0.16.1** 实测确认的两个默认模型：

- downbeat：`models/downbeats/2016/downbeats_blstm_1.pkl` → 输入 **314** 维，输出 **3** 维
- beat：`models/beats/2015/beats_blstm_1.pkl` → 输入 **266** 维，输出 **1** 维

> ⚠️ 两个易踩的坑（已实测）：
> 1. `beats_lstm_1.pkl`（2016，输入 162 维）是 **online** 配置（`frame_size=2048, num_bands=12`），
>    **不是**默认 beat 模型，不要误用。
> 2. downbeat 模型的 3 列输出顺序是 `[non-beat, beat, downbeat]`；madmom 用 `np.delete(obj=0)`
>    丢掉第 0 列，喂给 DBN 的是 `[beat, downbeat]`（即列 **1,2**）。

### 2. 导出 IR

```bash
cd D:\MusicRecorder
python .madmom_export/export_madmom_ir.py --model <downbeat.pkl> --fps 100 --out .madmom_export/out
# 可选：独立 beat 模型
python .madmom_export/export_madmom_ir.py --model <beats.pkl>   --fps 100 --out .madmom_export/out
```

产物写到 `.madmom_export/out/`：`ir.json`（拓扑 + 命名权重 base64 + meta）与 `madmom_weights.bin`
（CNN 常量占位，RNN 模型通常为空）。

> **数值自检（强烈建议）**：加 `--verify`，脚本会用纯 numpy 复刻的引擎对比 `nn.process` 的输出。
> 若 `max|ref-mine| < 1e-3` 即 `VERIFY PASS`，证明**门序 / 权重布局映射与 madmom 一致**；否则说明
> 转换器对当前模型结构的映射有偏差，需先修 `export_madmom_ir.py` 再继续。

### 3. 生成 C# 权重文件

```bash
python .madmom_export/gen_madmom_cs.py --ir .madmom_export/out/ir.json --cs Madmom/MadmomDownBeatIr.g.cs --class-name MadmomDownBeatIr
# 参数默认值：--ir .madmom_export/out/ir.json、--cs Madmom/MadmomIr.g.cs、--class-name MadmomIr
# beat 模型同理：--cs Madmom/MadmomBeatIr.g.cs --class-name MadmomBeatIr
```

生成 `Madmom/<类名>.g.cs`：算子常量 + 缓冲区元数据（时间维以 1 占位）+ 节点表 + **base64 内联的
命名权重** + `public static IrGraph Build()`。重新生成后直接重新编译即可，**无需改任何业务代码**。

---

## 构建与测试

```bash
dotnet build Madmom/Madmom.csproj            # netstandard2.1 库（不含权重文件也能编译）
dotnet run  --project Madmom.Test            # 跑冒烟测试（合成权重，不依赖真实模型）
```

`Madmom.Test` 用随机合成权重构造 `BLSTM→Dense→Sigmoid` 图，验证：
- 引擎动态 T 分配、RNN/Dense 前向、形状正确（[1]）；
- `DownBeatDbn` / `BeatTracker` 能跑通（[2][3]）；
- 构造的**周期拍点信号**能被 DBN 检出多拍（[4]，证明解码器对节奏敏感，而非随机噪声）。

`*Ir.g.cs` 缺失时，库本身仍可编译（权重在运行时由对应的 `Build()` 注入）；只有调用
`MadmomDownBeatIr.Build()` / `MadmomBeatIr.Build()` 时才需要该文件。注意 `Mp3ToSheet` 与
`Madmom.EndToEnd` 会直接调用它们，因此这两个工程需要权重文件在位（本仓库已内置）。

---

## 用法

```csharp
using Madmom;

// 两个 IR 均由生成步骤产出（见上）
IrGraph downbeatModel = MadmomDownBeatIr.Build();   // 314 → 3（[non-beat, beat, downbeat]）
IrGraph beatModel = MadmomBeatIr.Build();           // 266 → 1（beat），可选但推荐

var analyzer = new MadmomAnalyzer(downbeatModel, beatModel);
MadmomAnalyzer.Result r = analyzer.Analyze(samples, sampleRate);

foreach (var b in r.Downbeats)
    Console.WriteLine($"{b.Time:F3}s  beatInBar={b.BeatInBar}  bpb={b.BeatsPerBar}  bpm={b.Bpm}");
```

`MadmomAnalyzer.Analyze` 内部：读 `Fps`/`InputFeatureDim` → 特征提取 → 引擎前向 → 读 `OutputBuffer`
→ `DownBeatDbn.Track`（downbeat）与可选的 `BeatTracker.Track`（beat）。所有时间单位为秒。

---

## 架构与文件

| 文件 | 职责 |
|------|------|
| `IrGraph.cs` | IR 图描述：算子编码（0..26）、缓冲区元数据、命名权重 `Dictionary<string,float[]>`、模型元数据（InputFeatureDim / OutputDim / Fps / OutputBuffer）。 |
| `MadmomEngine.cs` | 纯 C# 前向引擎。**madmom 帧数 T 是动态的**（与 BasicPitch 固定输入不同），所有缓冲区在 `Run` 内按 `T` 重新分配，时间维前置到形状最前。支持 BasicPitch 全部 CNN 算子 + 新增 `Lstm(21)/Gru(22)/Blstm(23)/Dense(24)/Tanh(25)`。 |
| `Features.cs` | **忠实移植** madmom 特征管线：多分辨率（1024/2048/4096）STFT（原地迭代 FFT）→ `LogarithmicFilterbank` → `log10(spec+1)` → 正差分 hstack。带数已用 madmom 真实滤波组数值核对（downbeat 314 / beat 266）。 |
| `MadmomAnalyzer.cs` | 高层入口：音频 → 特征 → 引擎 → DBN。 |
| `Hmm.cs` | **忠实移植** `madmom/ml/hmm.pyx`：`TransitionModel`（CSR 稀疏，目标状态索引）、`ObservationModel`、`HiddenMarkovModel` + 对数域 **Viterbi**（含回溯）。 |
| `BeatsHmm.cs` | **忠实移植** `madmom/features/beats_hmm.py`：`BeatStateSpace`/`BarStateSpace`、`BeatTransitionModel`/`BarTransitionModel`、`exponential_transition`、两个 RNN 观测模型，以及 `DBNDownBeatTrackingProcessor` / `DBNBeatTrackingProcessor` 的 `process`（threshold 裁剪 → 多 HMM 选最佳 log_prob → peak 对齐）。 |
| `DownBeatDbn.cs` | `DBNDownBeatTrackingProcessor` 的薄封装，默认 `beats_per_bar=[3,4]`，MIN/MAX_BPM=55/215、NUM_TEMPI=60、TRANSITION_LAMBDA=100、OBSERVATION_LAMBDA=16、THRESHOLD=0.05、CORRECT=True。 |
| `BeatTracker.cs` | `DBNBeatTrackingProcessor` 的薄封装（NUM_TEMPI=None 线性、THRESHOLD=0、CORRECT=True）。 |
| `Madmom.Test/SmokeTest.cs` | 合成权重冒烟测试（不依赖真实模型）。 |
| `Madmom.EndToEnd/` | 端到端验证工程：WAV 解码 + 真实权重 + 全链路，含与 madmom 对照的状态空间硬校验。 |

### 算子编码（必须与 Python 侧完全一致）

```
0 Reshape   1 Slice     2 Pad       3 Unsqueeze  4 Conv      5 Neg       6 Transpose
7 Concat    8 Mul       9 ReduceSum 10 Sqrt     11 Add      12 Log      13 ReduceMin
14 Sub      15 ReduceMax 16 Div      17 Equal    18 Where    19 Relu     20 Sigmoid
21 Lstm     22 Gru      23 Blstm    24 Dense     25 Tanh     26 Softmax
```

### RNN / Dense 权重命名约定（引擎与生成器必须一致）

- 双向 BLSTM 节点 `idx`：`lstm_{idx}fx / lstm_{idx}fh / lstm_{idx}fb`（前向）、`...bx/bh/bb`（后向）；
  GRU 把前缀换成 `gru_`。
- 单向 LSTM/GRU 节点 `idx`：`lstm_{idx}x / lstm_{idx}h / lstm_{idx}b`（`gru_` 前缀）。
- Dense 节点 `idx`：`dense{idx}_x (out×in)` / `dense{idx}_b (out)`。

### 门序 / 布局映射（依据 madmom 源码 `madmom/ml/nn/layers.py`）

- **LSTM** 门序 `[input, forget, cell, output]`：每个 Gate 有 `weights(in,H)`、`recurrent(H,H)`、`bias(H,)`。
  引擎按 `[i,f,c,o]` 顺序乘加；`c = c_in*ig + state*fg`；输出 `tanh(state)*og`。
- **GRU** 门序 `[reset, update, cell]`：候选 `tanh(Wc·x + b + r⊙(Uc·h))`；输出 `u*cand + (1-u)*h`。
  引擎块序为 `[z(update), r(reset), n(candidate)]`，故转换器重排为 `[Wu, Wr, Wc]`。
- **BLSTM** 输出 `np.hstack((fwd, bwd[::-1]))`，即特征轴 `[前向; 后向]`。
- **Dense** `out = x·W(in,out) + b` → 引擎存 `W.T (out,in)`。

---

## 与 madmom 的对照验证（已完成）

早期文档曾把「特征提取」与「DBN 解码」标记为独立重写、待校准。两者现已**忠实移植并逐项对照通过**。

### 1. 特征提取

`Features.cs` 按 madmom 真实管线移植：`FramedSignal(fps=100)` → STFT → `LogarithmicFilterbank`
→ `log10(spec+1)` → `SpectrogramDifference(positive_diffs, hstack)` → 三分辨率 hstack。
用 madmom **真实** `LogarithmicFilterbank` 核对带数：downbeat → **314**、beat → **266**，
与两个模型的实测输入维一致 ✅。

### 2. DBN / HMM

对照基准是 madmom 自身源码：`.madmom_export/ref_dbn.py` 直接 import 官方 `features/beats_hmm.py`，
并按 `ml/hmm.pyx` 逐行转写 `TransitionModel` / `Viterbi`；两边喂**同一份激活**后比对：

| 校验项 | madmom 参考 | 本 C# 实现 |
|---|---|---|
| `BarStateSpace(3)` 状态数 | 11157 | 11157 ✅ |
| `BarStateSpace(4)` 状态数 | 14876 | 14876 ✅ |
| `BeatStateSpace` 状态数 / 间隔数 | 5617 / 82 | 5617 / 82 ✅ |
| downbeat 解码（8 拍，选中 4/4） | 0.390(1) 0.790(2) 1.190(3) 1.590(4) 1.990(1) 2.380(2) 2.780(3) 3.190(4) | **逐拍一致** ✅ |
| beat 解码（9 拍） | 0.000 0.390 0.790 1.190 1.590 1.990 2.390 2.790 3.190 | **逐拍一致** ✅ |

复现：

```bash
dotnet run --project Madmom.EndToEnd -- <audio.wav> --dump .madmom_export/act_dump
python .madmom_export/ref_dbn.py .madmom_export/act_dump 100
```

> `--verify`（导出脚本）校验的是**引擎前向数学**；上表校验的是**状态空间 + DBN 解码**。两者都通过。

### 可调参数

`DownBeatDbn(fps, tempoMin, tempoMax, beatsPerBar)` 与 `BeatTracker(fps, tempoMin, tempoMax)` 可覆盖默认
（默认值即 madmom 默认值）。要调更细的 `transition_lambda` / `observation_lambda` / `threshold`，
直接用 `BeatsHmm.cs` 中的 `DBNDownBeatTrackingProcessor` / `DBNBeatTrackingProcessor` 构造器。

---

## 性能

- 引擎为纯 C#（无 SIMD/Burst 时单段约数百毫秒级，BLSTM 为主）。移动端可仿照 `NmpBurstEngine`
  做 Burst + Job System 移植——引擎逐元素/规约/布局对并行友好。
- HMM 使用 CSR 稀疏转移，Viterbi 复杂度 `O(T · 转移数)`。downbeat 对 `[3,4]` 各跑一个 HMM 再选
  最佳 log_prob（与 madmom 一致）。实测 3.2 秒音频（320 帧）全链路约 0.7 秒（其中特征提取约 0.5 秒）。
- Viterbi 的回溯矩阵是 `T × num_states`，这是全链路最大的内存开销。已做两层优化：
  1. **回溯矩阵用 `ushort` 存储**（状态数 ≤ 65535），比 int 减半；
  2. **分块解码**（`MadmomAnalyzer.ChunkSeconds`，默认 10 秒 / 重叠 3 秒）：峰值内存只与
     块长有关，**与音频总长无关**。
- 分块规则：块 `i` 覆盖 `[start, start+L)`，`start` 步进 `L − overlap`；**重叠区结果取前一块**
  （前一块在该区已有完整上下文），后一块的开头 `overlap` 帧丢弃。采纳区首尾相接、无缝无重叠，
  故拍序列连续；拍号按「上一块末尾编号 +1」跨块递增，保证小节相位不乱。
  拍号（beats_per_bar）仍由**全局**累加 log_prob 选出，不是各块各选。
- 短于块长的音频不分块，**结果与 madmom 逐拍完全一致**（已回归验证）。
  设 `ChunkSeconds = 0` 可关闭分块（等价 madmom 一次性解码，内存随长度线性增长）。
- 特征阶段同样做了降内存改造（数值不变）：帧级临时缓冲 `frame/im/mag/filt` 复用而非逐帧
  `new`（4096 分辨率原本每帧分配 32 KB，19200 帧 ≈ 637 MB 的分配量）；log 谱改用长度
  `df+1` 的环形缓冲（正差分只需 `df` 帧历史）；各分辨率直接写入输出矩阵，省掉中间 `(T, 2·nb)`。

### 流式分块（特征 + RNN 前向 + DBN 全链路）

`MadmomAnalyzer.ChunkSeconds`（默认 10 s）同时驱动三段：

1. **特征**：`Features.ExtractBlock(audio, sr, cfg, frameStart, frameCount)` 只算这一段帧，
   按需重采样所需样本区间；块内多算 `df` 帧前缀提供差分历史、样本前后各留 `frame_size/2`
   上下文 —— **数值与一次性全量完全一致**（已 diff 验证）。
2. **RNN 前向**：`MadmomEngine.KeepState` 让 **前向** LSTM 的 h/c 跨块传递。
   BLSTM 的**后向**无法在线计算，每块从块末尾重新开始，这是唯一的近似来源。
3. **DBN**：见上一节。

激活只有 `T × 2` 个 float（192 秒约 150 KB），累积它不影响常量内存目标。

### 实测（192 秒音频，22050 Hz）

| 配置 | 峰值工作集 | 存活堆 | 与全量的差异 |
|---|---|---|---|
| 全量不分块 | 1460 MB | 64.7 MB | 基准 |
| 仅 DBN 分块 10 s | 469 MB | 64.7 MB | **逐拍一致** |
| **全流式 10 s** | **258 MB** | 64.7 MB | 480 拍中块边界附近 10 拍差 **1 帧（10 ms）** |

读法：**算法真实只需约 65 MB 存活对象**。峰值工作集是 .NET GC 的 committed 高水位
（GC 不把堆归还 OS），非算法需求；移动端 GC 更保守，靠「缩小块长 + 避免临时大数组」压低。

### 性能（同一 192 秒音频，Release）

| 配置 | 耗时 | 实时率 |
|---|---|---|
| Debug | 85.3 s | 2.3× |
| Release 标量（`MADMOM_NOSIMD=1`） | 39.1 s | 4.9× |
| **Release + SIMD（AVX2 8-wide）** | **24.1 s** | **8.0×** |

SIMD 用 `System.Numerics.Vector` 写在 `Simd.cs`，覆盖两处热点：特征滤波点积
（滤波矩阵转置为 `[band][bin]` 连续布局后才谈得上向量化）与 LSTM 的 `Wx·x + Wh·h`。
**ARM NEON 是同样的 128 位浮点 SIMD，Burst 会生成同类指令**，故本机的 1.62× 加速比
可直接外推到移动端（NEON 4-wide，收益略低，约 1.3×）。

### 移动端可行性结论

- **可以跑**，适合离线分析：桌面 8× 实时；按移动 CPU 单核约桌面 1/3、NEON 4-wide 估算，
  约 2× 实时（192 秒音频约 90 秒）。实际还要打折于发热降频与内存带宽。
- **内存**：流式 30 s 块下峰值 766 MB（GC 高水位），存活约 56 MB。缩小块长能压峰值，
  但**会显著劣化拍号精度**（见下），所以别盲目调小。
- **不适合**严格实时的跟随演奏场景（除非进一步做 Burst + Job System 并行与定点化）。

### 分块的精度代价（真实音乐实测，非合成循环音频）

参考 = `ChunkSeconds = 0`（不分块，与 madmom 逐拍等价）。素材：333 s 流行歌曲，477 个拍点。

| 块长 / 重叠 | 拍点 | 时刻完全一致 | **拍号一致率** | 孤立拍点 | 峰值内存 | 存活堆 |
|---|---|---|---|---|---|---|
| 不分块（参考） | 477 | 基准 | 基准 | 0 | 2301 MB | 2182 MB |
| **30 s / 3 s** | 480 | 89.3% | **80.3%** | 20 | 766 MB | 56 MB |
| 10 s / 3 s | 508 | 75.7% | 65.0% | 62 | 387 MB | 73 MB |

要点：

- **块长是第一位的精度旋钮**，重叠长度几乎不影响结果。实测 chunk=30 时把重叠从 3 s
  加到 8 s / 15 s，拍号一致率仍是 49% / 43%（3 s 时 41%）—— 说明孤立拍点**不是**
  后向预热不足造成的，加大重叠只是白白多算。
- **拍号错位曾会跨块累积**：原实现在块间接「+1 递增」续编号，某块一旦多检/漏检一拍，
  错位永久传播到后面所有块（实测一致率仅 40.7%）。改为「每块沿用自身解码相位、
  只在块边界对齐一次」后升到 80.3%，误差不再跨块传播。
- **建议块长 ≥ 30 s**。10 s 块虽然内存更省，但拍号一致率掉到 65%，孤立拍点 62 个。
  若要求与 madmom 逐拍完全一致，只能 `ChunkSeconds = 0`。

---

## 接入 GoFire 管线

`MadmomAnalyzer` 产出的每个 `Beat`（含 `BeatInBar` / `BeatsPerBar` / `Bpm`）可直接映射为 GoFire 的
`MusicalSegment`：每拍一个 segment，拍号与小节信息写入 segment 元数据，供 L3–L5 的节奏对齐使用。
这替代/补充了原本不准确的 L1 节奏输出，且全程在设备端运行（无需网络）。
