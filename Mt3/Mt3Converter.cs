using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Mt3
{
    /// <summary>
    /// MT3 multi-instrument music transcription converter.
    /// Uses MR-MT3 model (encoder + decoder ONNX) with log-mel spectrogram input.
    /// Segments are inferred in parallel; note decoding is sequential (stateful).
    /// </summary>
    public class Mt3Converter : IDisposable
    {
        private readonly InferenceSession _encoderSession;
        private readonly InferenceSession _decoderSession;
        private readonly int _maxParallelSegments;
        private bool _disposed;

        /// <summary>
        /// Create an MT3 converter from ONNX model file paths.
        /// </summary>
        /// <param name="encoderPath">Path to encoder ONNX model.</param>
        /// <param name="decoderPath">Path to decoder ONNX model.</param>
        /// <param name="maxParallelSegments">Max number of segments to infer in parallel. Defaults to CPU core count.</param>
        public Mt3Converter(string encoderPath, string decoderPath, int? maxParallelSegments = null)
        {
            var opts = CreateSessionOptions();
            _encoderSession = new InferenceSession(encoderPath, opts);
            _decoderSession = new InferenceSession(decoderPath, opts);
            _maxParallelSegments = ResolveMaxParallel(maxParallelSegments);
        }

        /// <summary>
        /// Create an MT3 converter from ONNX model bytes.
        /// Unity 中可通过 UnityWebRequest 读取 StreamingAssets 中的 ONNX 文件得到 byte[] 后传入，
        /// 避免 Android 等平台无法用 File API 直接读取 StreamingAssets 的问题。
        /// </summary>
        /// <param name="encoderBytes">Encoder ONNX model bytes.</param>
        /// <param name="decoderBytes">Decoder ONNX model bytes.</param>
        /// <param name="maxParallelSegments">Max number of segments to infer in parallel. Defaults to CPU core count.</param>
        public Mt3Converter(byte[] encoderBytes, byte[] decoderBytes, int? maxParallelSegments = null)
        {
            var opts = CreateSessionOptions();
            _encoderSession = new InferenceSession(encoderBytes, opts);
            _decoderSession = new InferenceSession(decoderBytes, opts);
            _maxParallelSegments = ResolveMaxParallel(maxParallelSegments);
        }

        private static SessionOptions CreateSessionOptions()
        {
            int cores = Math.Max(1, Environment.ProcessorCount);
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = Math.Min(4, cores)
            };
        }

        private static int ResolveMaxParallel(int? maxParallelSegments)
        {
            int cores = Math.Max(1, Environment.ProcessorCount);
            return maxParallelSegments ?? Math.Max(1, cores / 4);
        }

        /// <summary>
        /// 通过 <see cref="IAudioReader"/> 从文件读取音频并转换为音符列表。
        /// </summary>
        public List<Mt3Note> Convert(IAudioReader reader, string filePath, int maxDecodeSteps = 1024, IProgress<double> progress = null)
        {
            var (samples, sampleRate) = reader.Read(filePath);
            return Convert(samples, sampleRate, maxDecodeSteps, progress);
        }

        /// <summary>
        /// 通过 <see cref="IAudioReader"/> 读取文件并写入 MIDI。
        /// </summary>
        public void ConvertToMidi(IAudioReader reader, string filePath, string outputPath, int maxDecodeSteps = 1024, IProgress<double> progress = null)
        {
            var (samples, sampleRate) = reader.Read(filePath);
            ConvertToMidi(samples, sampleRate, outputPath, maxDecodeSteps, progress);
        }

        /// <summary>
        /// Convert audio samples to a list of notes.
        /// </summary>
        /// <param name="samples">Audio samples at original sample rate.</param>
        /// <param name="sampleRate">Original sample rate (will be resampled to 16kHz).</param>
        /// <param name="maxDecodeSteps">Maximum decoder steps per segment.</param>
        /// <param name="progress">Optional progress callback (0.0 - 1.0).</param>
        public List<Mt3Note> Convert(float[] samples, int sampleRate, int maxDecodeSteps = 1024, IProgress<double> progress = null)
        {
            // Resample to 16kHz and convert to mono
            float[] audio = ResampleToMono(samples, sampleRate, MelSpectrogram.SampleRate);

            // Preprocess to mel spectrogram batches
            var (melData, numSegments) = MelSpectrogram.Preprocess(audio);
            float segmentDuration = (float)MelSpectrogram.SegmentFrames / (MelSpectrogram.SampleRate / (float)MelSpectrogram.HopLength);

            // Step 1: Run encoder + decoder for all segments in parallel.
            // ONNX InferenceSession.Run is thread-safe, so segments share the sessions.
            var segmentTokens = new int[numSegments][];
            int completed = 0;
            var parallelOpts = new ParallelOptions { MaxDegreeOfParallelism = _maxParallelSegments };
            Parallel.For(0, numSegments, parallelOpts, s =>
            {
                segmentTokens[s] = InferSegment(melData, s, maxDecodeSteps);
                int done = Interlocked.Increment(ref completed);
                progress?.Report((double)done / numSegments);
            });

            // Step 2: Decode notes sequentially. The note-decoding state (active pitches,
            // current program, tied notes) carries over from one segment to the next,
            // so segments must be decoded in order.
            var state = new Mt3Codec.NoteDecodingState();
            for (int s = 0; s < numSegments; s++)
            {
                var tokens = segmentTokens[s];
                var codecTokens = new List<int>();
                foreach (int t in tokens)
                {
                    if (t == Mt3Codec.EosToken) break;
                    if (t < 3) continue; // skip PAD/EOS/UNK
                    codecTokens.Add(t - 3);
                }

                float startTime = s * segmentDuration;
                Mt3Codec.DecodeSegment(state, codecTokens.ToArray(), startTime, null);
            }

            return Mt3Codec.Flush(state);
        }

        /// <summary>
        /// Run encoder + greedy decoder for a single mel segment.
        /// Safe to call concurrently from multiple threads.
        /// </summary>
        private int[] InferSegment(float[] melData, int segmentIndex, int maxSteps)
        {
            int segOff = segmentIndex * MelSpectrogram.SegmentFrames * MelSpectrogram.NumMels;
            var segmentMel = new float[MelSpectrogram.SegmentFrames * MelSpectrogram.NumMels];
            Array.Copy(melData, segOff, segmentMel, 0, segmentMel.Length);

            // Encoder
            var encoderInput = new DenseTensor<float>(segmentMel, new[] { 1, MelSpectrogram.SegmentFrames, MelSpectrogram.NumMels });
            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("mel_input", encoderInput)
            };
            using var encResults = _encoderSession.Run(encoderInputs);
            float[] encHidden = encResults.First().AsTensor<float>().ToArray();

            // Decoder (autoregressive)
            return GreedyDecode(encHidden, maxSteps);
        }

        /// <summary>
        /// Greedy autoregressive decoding with KV-cache.
        /// Each step feeds only the new token; self-attention K/V is cached and reused.
        /// </summary>
        private int[] GreedyDecode(float[] encoderHidden, int maxSteps)
        {
            const int NumLayers = 8;
            const int NumHeads = 6;
            const int DKv = 64;

            var tokens = new List<int> { Mt3Codec.DecoderStartToken };
            int encSeqLen = encoderHidden.Length / 512;
            int currentToken = Mt3Codec.DecoderStartToken;

            // Past self-attention K/V cache per layer: shape [1, NumHeads, seqLen, DKv]
            float[][] pastK = new float[NumLayers][];
            float[][] pastV = new float[NumLayers][];

            var encoderInput = new DenseTensor<float>(encoderHidden, new[] { 1, encSeqLen, 512 });

            for (int step = 0; step < maxSteps; step++)
            {
                var decIds = new long[] { currentToken };
                var decoderInput = new DenseTensor<long>(decIds, new[] { 1, 1 });

                var decInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("decoder_input_ids", decoderInput),
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderInput)
                };

                // Add past self-attention K/V for each layer
                for (int i = 0; i < NumLayers; i++)
                {
                    int seqLen = pastK[i] != null ? pastK[i].Length / (NumHeads * DKv) : 0;
                    var kData = pastK[i] ?? Array.Empty<float>();
                    var vData = pastV[i] ?? Array.Empty<float>();
                    decInputs.Add(NamedOnnxValue.CreateFromTensor($"past_self_k_{i}",
                        new DenseTensor<float>(kData, new[] { 1, NumHeads, seqLen, DKv })));
                    decInputs.Add(NamedOnnxValue.CreateFromTensor($"past_self_v_{i}",
                        new DenseTensor<float>(vData, new[] { 1, NumHeads, seqLen, DKv })));
                }

                using var decResults = _decoderSession.Run(decInputs);
                var outputs = decResults.ToArray();

                // logits output: [batch=1, seq=1, vocab]
                var logits = outputs[0].AsTensor<float>();
                int vocabSize = (int)logits.Dimensions[2];
                float[] logitArray = logits.ToArray();

                int bestToken = 0;
                float bestLogit = float.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                {
                    if (logitArray[v] > bestLogit)
                    {
                        bestLogit = logitArray[v];
                        bestToken = v;
                    }
                }

                tokens.Add(bestToken);
                currentToken = bestToken;

                // Update K/V cache: outputs[0]=logits, then per layer: self_k, self_v, cross_k, cross_v
                for (int i = 0; i < NumLayers; i++)
                {
                    int off = 1 + i * 4;
                    pastK[i] = outputs[off].AsTensor<float>().ToArray();
                    pastV[i] = outputs[off + 1].AsTensor<float>().ToArray();
                }

                if (bestToken == Mt3Codec.EosToken) break;
            }

            return tokens.Skip(1).ToArray();
        }

        /// <summary>
        /// Convert notes to MIDI file.
        /// </summary>
        public void ConvertToMidi(float[] samples, int sampleRate, string outputPath, int maxDecodeSteps = 1024, IProgress<double> progress = null)
        {
            var notes = Convert(samples, sampleRate, maxDecodeSteps, progress);
            Mt3MidiWriter.Write(notes, outputPath);
        }

        private static float[] ResampleToMono(float[] samples, int srcRate, int dstRate)
        {
            // Simple linear resampling + mono mix
            int srcLen = samples.Length;
            int dstLen = (int)(srcLen * (long)dstRate / srcRate);
            var result = new float[dstLen];
            for (int i = 0; i < dstLen; i++)
            {
                double srcPos = i * (double)srcRate / dstRate;
                int idx = (int)srcPos;
                double frac = srcPos - idx;
                float a = idx < srcLen ? samples[idx] : 0;
                float b = idx + 1 < srcLen ? samples[idx + 1] : 0;
                result[i] = (float)(a * (1 - frac) + b * frac);
            }
            return result;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _encoderSession.Dispose();
                _decoderSession.Dispose();
                _disposed = true;
            }
        }
    }
}
