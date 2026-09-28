using System;
using System.IO;
using System.Reflection;

namespace Mt3
{
    /// <summary>
    /// Log-mel spectrogram computation matching torchaudio MelSpectrogram
    /// (n_fft=2048, hop=128, n_mels=512, f_min=20, f_max=7600, power=1.0, center=False).
    /// </summary>
    public static class MelSpectrogram
    {
        public const int SampleRate = 16000;
        public const int FftSize = 2048;
        public const int HopLength = 128;
        public const int NumMels = 512;
        public const float FMin = 20.0f;
        public const float FMax = 7600.0f;
        public const int NumFreqBins = FftSize / 2 + 1; // 1025
        public const int SegmentFrames = 256; // 256 frames per segment
        public const int SegmentSamples = SegmentFrames * HopLength; // 32768

        private static readonly float[] _melFilterbank; // [NumFreqBins * NumMels]
        private static readonly float[] _window; // [FftSize]

        static MelSpectrogram()
        {
            _melFilterbank = LoadEmbedded("mel_filterbank.bin");
            _window = LoadEmbedded("mel_window.bin");
        }

        private static float[] LoadEmbedded(string name)
        {
            var asm = typeof(MelSpectrogram).Assembly;
            // Resource names are prefixed with the root namespace
            string fullName = $"Mt3.{name}";
            using var stream = asm.GetManifestResourceStream(fullName);
            if (stream == null)
                throw new InvalidOperationException($"Embedded resource '{fullName}' not found.");
            var bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }

        /// <summary>
        /// Compute log-mel spectrogram for a single audio segment.
        /// </summary>
        /// <param name="samples">Audio samples (length should be SegmentSamples for full segment).</param>
        /// <returns>Log-mel spectrogram as [time_frames, NumMels].</returns>
        public static float[] ComputeSegment(float[] samples)
        {
            int nSamples = samples.Length;
            // pad_end: zero-pad end to ensure complete STFT frames (matches MR-MT3)
            int nFramesCeil = (nSamples + HopLength - 1) / HopLength;
            int padSamples = Math.Max(0, FftSize + HopLength * (nFramesCeil - 1) - nSamples);
            int paddedLen = nSamples + padSamples;
            var padded = new float[paddedLen];
            Array.Copy(samples, padded, nSamples);

            // Number of STFT frames (center=False)
            int numFrames = (paddedLen - FftSize) / HopLength + 1;

            // Apply window and compute magnitude spectrogram per frame
            // magnitude[freq, frame]
            var magnitude = new float[NumFreqBins * numFrames];
            var frameBuf = new float[FftSize];
            var re = new float[FftSize];
            var im = new float[FftSize];

            for (int f = 0; f < numFrames; f++)
            {
                int offset = f * HopLength;
                for (int i = 0; i < FftSize; i++)
                {
                    frameBuf[i] = padded[offset + i] * _window[i];
                }
                FFT(frameBuf, re, im);
                for (int k = 0; k < NumFreqBins; k++)
                {
                    float mag = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
                    magnitude[k * numFrames + f] = mag;
                }
            }

            // Apply mel filterbank: mel[mel_bin, frame] = sum_k magnitude[k, frame] * fb[k, mel_bin]
            var mel = new float[NumMels * numFrames];
            for (int m = 0; m < NumMels; m++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    float sum = 0;
                    for (int k = 0; k < NumFreqBins; k++)
                    {
                        sum += magnitude[k * numFrames + f] * _melFilterbank[k * NumMels + m];
                    }
                    mel[m * numFrames + f] = sum;
                }
            }

            // Take log (safe) and transpose to [time, mel]
            var result = new float[numFrames * NumMels];
            for (int f = 0; f < numFrames; f++)
            {
                for (int m = 0; m < NumMels; m++)
                {
                    float v = mel[m * numFrames + f];
                    if (v <= 0) v = 1e-5f;
                    result[f * NumMels + m] = MathF.Log(v);
                }
            }

            return result;
        }

        /// <summary>
        /// In-place radix-2 Cooley-Tukey FFT.
        /// </summary>
        private static void FFT(float[] input, float[] re, float[] im)
        {
            int n = input.Length;
            Array.Copy(input, re, n);
            Array.Clear(im, 0, n);

            // Bit reversal
            int j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                while (j >= bit)
                {
                    j -= bit;
                    bit >>= 1;
                }
                j += bit;
                if (i < j)
                {
                    (re[i], re[j]) = (re[j], re[i]);
                    (im[i], im[j]) = (im[j], im[i]);
                }
            }

            // Butterfly
            for (int len = 2; len <= n; len <<= 1)
            {
                float ang = -2f * MathF.PI / len;
                float wlenRe = MathF.Cos(ang);
                float wlenIm = MathF.Sin(ang);
                for (int i = 0; i < n; i += len)
                {
                    float wRe = 1f, wIm = 0f;
                    int half = len >> 1;
                    for (int k = 0; k < half; k++)
                    {
                        float uRe = re[i + k];
                        float uIm = im[i + k];
                        float vRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
                        float vIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
                        re[i + k] = uRe + vRe;
                        im[i + k] = uIm + vIm;
                        re[i + k + half] = uRe - vRe;
                        im[i + k + half] = uIm - vIm;
                        float newWRe = wRe * wlenRe - wIm * wlenIm;
                        wIm = wRe * wlenIm + wIm * wlenRe;
                        wRe = newWRe;
                    }
                }
            }
        }

        /// <summary>
        /// Preprocess audio into model input batches: (numSegments, SegmentFrames, NumMels).
        /// </summary>
        /// <param name="samples">Audio at 16kHz.</param>
        /// <returns>Flattened float array of shape [numSegments, SegmentFrames, NumMels].</returns>
        public static (float[] data, int numSegments) Preprocess(float[] samples)
        {
            // Split into frames of HopLength samples
            int numFrames = samples.Length / HopLength;
            int numSegments = (int)Math.Ceiling((double)numFrames / SegmentFrames);
            if (numSegments == 0) numSegments = 1;

            var allMel = new float[numSegments * SegmentFrames * NumMels];

            for (int s = 0; s < numSegments; s++)
            {
                int startFrame = s * SegmentFrames;
                int endFrame = Math.Min(startFrame + SegmentFrames, numFrames);
                int length = endFrame - startFrame;

                // Build segment audio: always SegmentSamples (32768) samples, zero-padded
                var segment = new float[SegmentSamples];
                for (int f = 0; f < length; f++)
                {
                    int srcOff = (startFrame + f) * HopLength;
                    int dstOff = f * HopLength;
                    Array.Copy(samples, srcOff, segment, dstOff, HopLength);
                }

                // Compute mel spectrogram for this segment
                var mel = ComputeSegment(segment); // [timeFrames, NumMels]

                // Only copy real frames (0..length-1); padding frames stay at 0.0
                // (matching Python which zeros out spectrograms[i, length:, :] = 0)
                int framesToCopy = Math.Min(length, SegmentFrames);
                for (int f = 0; f < framesToCopy; f++)
                {
                    int dstOff = (s * SegmentFrames + f) * NumMels;
                    int srcOff = f * NumMels;
                    Array.Copy(mel, srcOff, allMel, dstOff, NumMels);
                }
            }

            return (allMel, numSegments);
        }
    }
}
