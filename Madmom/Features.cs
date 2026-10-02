using System;
using System.Collections.Generic;

namespace Madmom;

/// <summary>
/// 忠实移植 madmom 的 <c>RNNDownBeatProcessor</c> / <c>RNNBeatProcessor</c> 特征管线（C# 纯实现，运行时不依赖 Python）。
///
/// 输入约定：单声道 float32 PCM，归一化到 [-1, 1]，任意采样率（内部重采样到 44100）。
/// 这与 madmom 的 <c>SignalProcessor(num_channels=1, sample_rate=44100)</c> 后接 STFT 等价：
/// madmom 对 int16 输入会把 Hann 窗按 1/32768 缩放，而这恰好等于“把音频除以 32768 得到
/// [-1,1] float 后再用不带缩放的 Hann 窗”——所以本实现直接吃 [-1,1] float + 原始 Hann 窗，逐帧数值一致。
///
/// 管线（每个分辨率 frame_size ∈ {1024,2048,4096}，bands_per_octave 见配置）：
///   1. FramedSignalProcessor(frame_size, fps=100) → hop = sampleRate/100（44100 下 = 441）
///   2. STFT：Hann 窗 * 帧 → rfft，取前 frame_size/2 个 bin 的幅度
///   3. FilteredSpectrogramProcessor(LogarithmicFilterbank, fmin=30, fmax=17000, norm=True)
///   4. LogarithmicSpectrogramProcessor(mul=1, add=1) → log10(spec+1)
///   5. SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=hstack)
///      → 每帧特征 = [logSpec, max(0, logSpec[t]-logSpec[t-diff_frames])]，共 2*num_bands
///   6. 三个分辨率在频率维 hstack → 总维 = 2*(b0+b1+b2)，b_i 由各分辨率实际生成的三角滤波带数决定
///      （madmom 的 num_bands 是「每倍频程带数」，不是带数本身）
/// </summary>
public static class Features
{
    public const double Fps = 100.0;
    public const double FMin = 30.0;
    public const double FMax = 17000.0;
    public const double FRef = 440.0;
    public const int TargetSampleRate = 44100;

    private const double DiffRatio = 0.5;

    // ----------------------------------------------------------------- 配置
    public sealed class FeatureConfig
    {
        /// <summary>每个分辨率：[frame_size, bands_per_octave]。</summary>
        public int[][] Resolutions { get; }

        public FeatureConfig(int[][] resolutions) => Resolutions = resolutions;

        /// <summary>downbeat 模型特征：frame_size=[1024,2048,4096]，bands/倍频程=[3,6,12]。</summary>
        public static readonly FeatureConfig DownBeat = new(new int[][]
        {
            new[] { 1024, 3 }, new[] { 2048, 6 }, new[] { 4096, 12 },
        });

        /// <summary>beat 模型特征：三个分辨率都用 6 带/倍频程。</summary>
        public static readonly FeatureConfig Beat = new(new int[][]
        {
            new[] { 1024, 6 }, new[] { 2048, 6 }, new[] { 4096, 6 },
        });
    }

    // ------------------------------------------------- 公开计划（供外部引擎复用）
    //
    // Unity Burst 内核（见 gofire Standard Assets/Madmom/MadmomBurstEngine.cs）需要与托管
    // ExtractBlock 完全相同的滤波带 / 窗数据与分块编排。这里把三者公开：
    //   ResolutionPlan —— 单分辨率的 Hann 窗与对数滤波带（平铺 float，Burst 友好）；
    //   BuildPlan      —— 一个 FeatureConfig 的全部分辨率计划（数值与内部缓存完全一致）；
    //   GetBlockSpan   —— 分块边界推导（前缀帧数 dfMax、样本留白 halfMax，与托管实现同源）。
    // ResampleRange 同样公开供外部按 span 重采样。

    /// <summary>单分辨率计划：窗 + 滤波带 + 差分历史帧数。</summary>
    public sealed class ResolutionPlan
    {
        public int FrameSize;
        public int NumBands;
        public int DiffFrames;
        public int Half;
        public float[] Hann;   // [FrameSize]
        public float[] FbT;    // [NumBands * Half]，band 主序平铺（第 b 带占 [b*Half, (b+1)*Half)）
    }

    /// <summary>构建一个 FeatureConfig 的全部分辨率计划（数据取自内部缓存，逐值一致）。</summary>
    public static ResolutionPlan[] BuildPlan(FeatureConfig cfg)
    {
        var plans = new ResolutionPlan[cfg.Resolutions.Length];
        for (int r = 0; r < cfg.Resolutions.Length; r++)
        {
            var p = Prepare(cfg.Resolutions[r][0], cfg.Resolutions[r][1]);
            int half = p.FrameSize / 2;
            var flat = new float[p.NumBands * half];
            for (int b = 0; b < p.NumBands; b++)
                for (int i = 0; i < half; i++)
                    flat[b * half + i] = p.FbT[b][i];
            var hann = new float[p.FrameSize];
            for (int i = 0; i < p.FrameSize; i++)
                hann[i] = (float)p.Hann[i];   // 与托管路径逐帧乘的 (float)Hann[i] 同值
            plans[r] = new ResolutionPlan
            {
                FrameSize = p.FrameSize,
                NumBands = p.NumBands,
                DiffFrames = p.DiffFrames,
                Half = half,
                Hann = hann,
                FbT = flat,
            };
        }
        return plans;
    }

    /// <summary>分块边界：From=含差分前缀的帧起点，Lo/Hi=样本区间 [Lo,Hi)，T=含前缀的总帧数。</summary>
    public struct BlockSpan
    {
        public int From, Lo, Hi, T;
    }

    /// <summary>推导分块边界（与 ExtractBlock 内部逻辑同源；Unity 侧用它保证跨块数值一致）。</summary>
    public static BlockSpan GetBlockSpan(FeatureConfig cfg, int sampleRate, int audioLength,
        int frameStart, int frameCount)
    {
        int dfMax = 0, halfMax = 0;
        for (int r = 0; r < cfg.Resolutions.Length; r++)
        {
            var p = Prepare(cfg.Resolutions[r][0], cfg.Resolutions[r][1]);
            if (p.DiffFrames > dfMax) dfMax = p.DiffFrames;
            if (p.FrameSize / 2 > halfMax) halfMax = p.FrameSize / 2;
        }
        double hop = TargetSampleRate / Fps; // 441.0
        var s = new BlockSpan();
        s.From = Math.Max(0, frameStart - dfMax);
        int n441 = sampleRate == TargetSampleRate
            ? audioLength
            : (int)Math.Ceiling(audioLength * (double)TargetSampleRate / sampleRate);
        s.Lo = Math.Max(0, (int)Math.Round(s.From * hop) - halfMax);
        s.Hi = Math.Min(n441, (int)Math.Round((frameStart + frameCount - 1) * hop) + halfMax + 1);
        s.T = frameStart + frameCount - s.From;
        return s;
    }

    // ------------------------------------------------------- 预计算的分辨率数据
    private sealed class Prepared
    {
        public int FrameSize;
        public float[][] Fb;      // [numBins][numBands]
        /// <summary>
        /// Fb 的转置 [numBands][numBins]：滤波本质是「每个带对频谱做一次点积」，
        /// 转置后点积沿连续内存走，cache 友好，也才能向量化（原布局是跨 stride 列访问）。
        /// </summary>
        public float[][] FbT;
        public int NumBands;
        public int DiffFrames;
        public double[] Hann;     // [frameSize]
    }

    private sealed class FilterBand
    {
        public int Start;
        public float[] Data;   // 长度 = stop - start
    }

    private static readonly Dictionary<string, Prepared> _cache = new();

    private static Prepared Prepare(int frameSize, int bandsPerOctave)
    {
        string key = frameSize + "_" + bandsPerOctave;
        lock (_cache)
        {
            if (_cache.TryGetValue(key, out var p)) return p;
            p = BuildPrepared(frameSize, bandsPerOctave);
            _cache[key] = p;
            return p;
        }
    }

    private static Prepared BuildPrepared(int frameSize, int bandsPerOctave)
    {
        int numFftBins = frameSize / 2;
        double[] binFreq = new double[numFftBins];
        for (int i = 0; i < numFftBins; i++)
            binFreq[i] = (double)i * TargetSampleRate / frameSize;

        float[][] fb = BuildLogFilterbank(binFreq, bandsPerOctave, out int numBands);

        double[] hann = new double[frameSize];
        for (int i = 0; i < frameSize; i++)
            hann[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (frameSize - 1));

        int diffFrames = DiffFrames(DiffRatio, TargetSampleRate / Fps, frameSize, hann);

        // 转置：FbT[band] 是该带在全部 bin 上的权重（连续），供点积使用
        var fbT = new float[numBands][];
        for (int b = 0; b < numBands; b++)
        {
            var row = new float[numFftBins];
            for (int i = 0; i < numFftBins; i++) row[i] = fb[i][b];
            fbT[b] = row;
        }

        return new Prepared
        {
            FrameSize = frameSize,
            Fb = fb,
            FbT = fbT,
            NumBands = numBands,
            DiffFrames = diffFrames,
            Hann = hann,
        };
    }

    // ------------------------------------------------ LogarithmicFilterbank 移植
    // 等价于 madmom: LogarithmicFilterbank(binFreq, num_bands=bandsPerOctave, fmin, fmax,
    //                                      fref=440, norm_filters=True, unique_filters=True, bands_per_octave=True)
    // 返回 [numBins][numBands]（列即带）。
    private static float[][] BuildLogFilterbank(double[] binFreq, int bandsPerOctave, out int numBands)
    {
        // 1) log_frequencies(num_bands, fmin, fmax, fref)
        double logMin = Math.Log(FMin / FRef, 2.0) * bandsPerOctave;
        double logMax = Math.Log(FMax / FRef, 2.0) * bandsPerOctave;
        int left = (int)Math.Floor(logMin);
        int right = (int)Math.Ceiling(logMax);
        int m = right - left;
        double[] freqs = new double[m];
        for (int i = 0; i < m; i++)
            freqs[i] = FRef * Math.Pow(2.0, (left + i) / (double)bandsPerOctave);

        // 2) 按 fmin / fmax 裁剪（searchsorted left / right）
        int s0 = SearchSortedLeft(freqs, FMin);
        int s1 = SearchSortedRight(freqs, FMax);
        double[] freqsTrim = new double[Math.Max(0, s1 - s0)];
        for (int i = 0; i < freqsTrim.Length; i++) freqsTrim[i] = freqs[s0 + i];

        // 3) frequencies2bins(freqsTrim, binFreq, unique_bins=True)
        int[] bins = Frequencies2Bins(freqsTrim, binFreq, uniqueBins: true);

        // 4) 构造重叠三角滤波带（保留 start）
        var bands = TriangularFilterBands(bins, norm: true);

        // 5) Filterbank.from_filters → [numBins][numBands]
        float[][] fb = FromFilters(binFreq.Length, bands);
        numBands = fb.Length == 0 ? 0 : fb[0].Length;
        return fb;
    }

    private static int[] Frequencies2Bins(double[] frequencies, double[] binFreq, bool uniqueBins)
    {
        int n = binFreq.Length;
        int[] indices = new int[frequencies.Length];
        for (int k = 0; k < frequencies.Length; k++)
        {
            int idx = SearchSortedLeft(binFreq, frequencies[k]);
            if (idx < 1) idx = 1;
            else if (idx > n - 1) idx = n - 1;
            double leftV = binFreq[idx - 1];
            double rightV = binFreq[idx];
            if (frequencies[k] - leftV < rightV - frequencies[k]) idx -= 1;
            indices[k] = idx;
        }
        if (uniqueBins)
        {
            var seen = new List<int>();
            var outIdx = new List<int>();
            foreach (int v in indices)
            {
                if (!seen.Contains(v)) { seen.Add(v); outIdx.Add(v); }
            }
            return outIdx.ToArray();
        }
        return indices;
    }

    /// <summary>TriangularFilter.filters(bins, norm, overlap=True)：从连续三元组 (start,center,stop) 造三角滤波带并保留 start。</summary>
    private static List<FilterBand> TriangularFilterBands(int[] bins, bool norm)
    {
        var bands = new List<FilterBand>();
        int index = 0;
        while (index + 3 <= bins.Length)
        {
            int start = bins[index];
            int center = bins[index + 1];
            int stop = bins[index + 2];
            if (stop - start < 2) { center = start; stop = start + 1; }
            float[] data = TriangularFilterData(start, ref center, ref stop, norm);
            bands.Add(new FilterBand { Start = start, Data = data });
            index++;
        }
        return bands;
    }

    private static float[] TriangularFilterData(int start, ref int center, ref int stop, bool norm)
    {
        center -= start;
        stop -= start;
        float[] data = new float[stop];
        for (int i = 0; i < center; i++)
            data[i] = (float)((double)i / center);                 // linspace(0,1,center,endpoint=False)
        int fallLen = stop - center;
        for (int i = 0; i < fallLen; i++)
            data[center + i] = (float)(1.0 - (double)i / fallLen);  // linspace(1,0,fallLen,endpoint=False)
        if (norm)
        {
            double sum = 0;
            foreach (double v in data) sum += v;
            if (sum != 0) for (int i = 0; i < data.Length; i++) data[i] = (float)(data[i] / sum);
        }
        return data;
    }

    /// <summary>Filterbank.from_filters：把每个滤波带放到 [numBins][numBands]，重叠处取 max（_put_filter）。</summary>
    private static float[][] FromFilters(int numFftBins, List<FilterBand> bands)
    {
        float[][] fb = new float[numFftBins][];
        for (int i = 0; i < numFftBins; i++) fb[i] = new float[bands.Count];
        for (int b = 0; b < bands.Count; b++)
        {
            FilterBand fbnd = bands[b];
            int s = fbnd.Start;
            int e = s + fbnd.Data.Length;
            int cs = Math.Max(0, s);
            int ce = Math.Min(numFftBins, e);
            for (int i = cs; i < ce; i++)
            {
                float v = fbnd.Data[i - s];
                if (v > fb[i][b]) fb[i][b] = v; // 重叠取 max
            }
        }
        return fb;
    }

    // --------------------------------------------- diff_frames（SpectrogramDifference._diff_frames）
    private static int DiffFrames(double diffRatio, double hopSize, int frameSize, double[] window)
    {
        double maxW = 0;
        foreach (double w in window) if (w > maxW) maxW = w;
        int sample = 0;
        for (int i = 0; i < frameSize; i++)
        {
            if (window[i] > diffRatio * maxW) { sample = i; break; }
        }
        double diffSamples = frameSize / 2.0 - sample;
        return (int)Math.Max(1, Math.Round(diffSamples / hopSize));
    }

    // ----------------------------------------------------------- 主提取入口
    /// <summary>
    /// 从音频提取特征。返回 (frames, featureDim)。featureDim 由各分辨率实际带数动态决定。
    /// </summary>
    /// <summary>总帧数（与一次性 Extract 的行数一致）。</summary>
    public static int FrameCount(float[] audio, int sampleRate)
    {
        int n441 = sampleRate == TargetSampleRate
            ? audio.Length
            : (int)Math.Ceiling(audio.Length * (double)TargetSampleRate / sampleRate);
        return (int)Math.Ceiling(n441 / (TargetSampleRate / Fps));
    }

    /// <summary>一次性提取全部帧（等价于 ExtractBlock(0, FrameCount)）。</summary>
    public static float[][] Extract(float[] audio, int sampleRate, FeatureConfig cfg, out int featureDim)
        => ExtractBlock(audio, sampleRate, cfg, 0, FrameCount(audio, sampleRate), out featureDim);

    /// <summary>
    /// 分块提取：只计算 [frameStart, frameStart+frameCount) 这些帧。样本按需从 audio 重采样，
    /// 所有中间产物均为 O(frameCount)，与音频总长无关。数值与一次性全量完全一致。
    /// </summary>
    public static float[][] ExtractBlock(float[] audio, int sampleRate, FeatureConfig cfg,
        int frameStart, int frameCount, out int featureDim)
    {
        if (frameCount <= 0) { featureDim = 0; return Array.Empty<float[]>(); }

        double hop = TargetSampleRate / Fps; // 441.0
        var prepared = new Prepared[cfg.Resolutions.Length];
        for (int r = 0; r < cfg.Resolutions.Length; r++)
            prepared[r] = Prepare(cfg.Resolutions[r][0], cfg.Resolutions[r][1]);

        // 正差分需要 df 帧历史、STFT 窗口需要 frame_size/2 样本：块内多算 df 帧前缀（不输出），
        // 样本区间前后各留 halfMax，保证跨块数值与全量一致（边界推导见 GetBlockSpan）。
        var span = GetBlockSpan(cfg, sampleRate, audio.Length, frameStart, frameCount);
        int from = span.From;

        float[] mono = ResampleRange(audio, sampleRate, span.Lo, span.Hi);
        int N = mono.Length;
        int T = span.T;                           // 含不输出的前缀帧

        featureDim = 0;
        for (int r = 0; r < cfg.Resolutions.Length; r++)
            featureDim += 2 * prepared[r].NumBands;

        // 内存优化（长音频下这些是 GC 高水位主因，数值与优化前完全一致）：
        //  1. 帧级临时缓冲（frame/im/mag/filt）复用，不逐帧 new —— 4096 分辨率原本每帧要
        //     分配 32KB，19200 帧 ≈ 637MB 的分配量。
        //  2. log 谱改用长度 df+1 的环形缓冲（正差分只需要 df 帧历史），不再存整条 T×nb。
        //  3. 各分辨率直接写入 features 的对应列区间，省掉中间 (T, 2*nb) 的 blocks。
        var features = new float[T][];
        for (int t = 0; t < T; t++) features[t] = new float[featureDim];

        int offBase = 0;
        for (int r = 0; r < cfg.Resolutions.Length; r++)
        {
            Prepared p = prepared[r];
            int fs = p.FrameSize;
            int nb = p.NumBands;
            int half = fs / 2;
            int df = Math.Max(0, p.DiffFrames);
            int off = offBase;
            offBase += 2 * nb;

            var frame = new float[fs];
            var im = new float[fs];
            var mag = new float[half];
            var filt = new float[nb];
            var ring = new float[df + 1][];
            for (int k = 0; k <= df; k++) ring[k] = new float[nb];
            int cur = 0;

            for (int t = 0; t < T; t++)
            {
                // 帧 (from+t) 的中心样本，换算到本块 mono 的局部坐标
                int refSamp = (int)Math.Round((from + t) * hop) - span.Lo;
                int start = refSamp - half; // frame_size//2, origin=0
                for (int i = 0; i < fs; i++)
                {
                    int idx = start + i;
                    frame[i] = (idx >= 0 && idx < N) ? mono[idx] : 0f;
                }
                for (int i = 0; i < fs; i++) frame[i] *= (float)p.Hann[i];
                Array.Clear(im, 0, fs);
                Fft(frame, im, false);
                for (int i = 0; i < half; i++)
                    mag[i] = (float)Math.Sqrt((double)frame[i] * frame[i] + (double)im[i] * im[i]);
                // 每个带与频谱做一次点积（转置布局 + SIMD，这是特征阶段最重的一步）
                for (int b = 0; b < nb; b++)
                    filt[b] = (float)Simd.DotDouble(p.FbT[b], 0, mag, 0, half);
                var ls = ring[cur];
                for (int b = 0; b < nb; b++) ls[b] = (float)Math.Log10(filt[b] + 1.0);

                var row = features[t];
                for (int b = 0; b < nb; b++) row[off + b] = ls[b];
                if (t >= df)
                {
                    var prev = ring[(cur - df + df + 1) % (df + 1)];
                    for (int b = 0; b < nb; b++)
                    {
                        float v = ls[b] - prev[b];
                        row[off + nb + b] = v > 0f ? v : 0f;
                    }
                }
                // t < df 时差分列保持 0（数组已初始化为 0），与原实现一致
                cur = (cur + 1) % (df + 1);
            }
        }

        int drop = frameStart - from; // 丢掉只为提供差分历史而多算的前缀帧
        if (drop == 0) return features;
        var outp = new float[frameCount][];
        Array.Copy(features, drop, outp, 0, frameCount);
        return outp;
    }

    // --------------------------------------------------------------- 重采样（线性，44100 为精确 no-op）
    // madmom 用 ffmpeg (FFT) 重采样；此处对 44100 输入为精确 no-op，对非 44100 用线性插值近似。
    // 默认与测试均使用 44100 输入，故逐帧数值与 madmom 一致。
    /// <summary>
    /// 只重采样 [outLo, outHi) 这段（44100 域）。线性插值只需 ±1 样本上下文，故可分块调用；
    /// 44100 输入且区间覆盖全段时零拷贝。
    /// </summary>
    public static float[] ResampleRange(float[] audio, int sampleRate, int outLo, int outHi)
    {
        if (sampleRate == TargetSampleRate)
        {
            if (outLo == 0 && outHi >= audio.Length) return audio; // no-op，零拷贝
            int len = Math.Max(0, Math.Min(outHi, audio.Length) - outLo);
            var slice = new float[len];
            if (len > 0) Array.Copy(audio, outLo, slice, 0, len);
            return slice;
        }
        double ratio = (double)TargetSampleRate / sampleRate;
        int outLen = Math.Max(0, outHi - outLo);
        float[] outp = new float[outLen];
        int last = audio.Length - 1;
        for (int i = 0; i < outLen; i++)
        {
            double pos = (outLo + i) / ratio;
            int i0 = (int)pos;
            if (i0 < 0) i0 = 0; else if (i0 > last) i0 = last;
            int i1 = Math.Min(i0 + 1, last);
            double frac = pos - i0;
            outp[i] = (float)(audio[i0] * (1.0 - frac) + audio[i1] * frac);
        }
        return outp;
    }

    // ----------------------------------------------------------------- 通用数学
    private static int SearchSortedLeft(double[] sorted, double v)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (sorted[mid] < v) lo = mid + 1; else hi = mid; }
        return lo;
    }

    private static int SearchSortedRight(double[] sorted, double v)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (sorted[mid] <= v) lo = mid + 1; else hi = mid; }
        return lo;
    }

    /// <summary>原地迭代 FFT（Cooley-Tukey，长度须为 2 的幂）。re/im 同长。</summary>
    public static void Fft(float[] re, float[] im, bool inverse)
    {
        int n = re.Length;
        if ((n & (n - 1)) != 0) throw new ArgumentException("FFT 长度须为 2 的幂");
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = (inverse ? 2 : -2) * Math.PI / len;
            double wr = Math.Cos(ang), wi = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double cr = 1, ci = 0;
                for (int k = 0; k < len / 2; k++)
                {
                    int a = i + k, b = i + k + len / 2;
                    double xr = re[a] + cr * re[b] - ci * im[b];
                    double xi = im[a] + cr * im[b] + ci * re[b];
                    double yr = re[a] - cr * re[b] + ci * im[b];
                    double yi = im[a] - cr * im[b] - ci * re[b];
                    re[a] = (float)xr; im[a] = (float)xi; re[b] = (float)yr; im[b] = (float)yi;
                    double ncr = cr * wr - ci * wi;
                    double nci = cr * wi + ci * wr;
                    cr = ncr; ci = nci;
                }
            }
        }
        if (inverse)
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}
