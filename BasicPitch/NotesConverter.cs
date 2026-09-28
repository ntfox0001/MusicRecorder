using System;
using System.Collections.Generic;
using System.Linq;

namespace BasicPitch;

/// <summary>
/// 音符转换参数。
/// </summary>
public record struct NotesConvertOptions
{
    public float OnsetThreshold = 0.5f;
    public float FrameThreshold = 0.3f;
    public int MinNoteLength = 11;
    public int EnergyThreshold = 11;
    public float? MinFreq = null;
    public float? MaxFreq = null;
    public bool InferOnsets = true;
    public bool IncludePitchBends = true;
    public bool MelodiaTrick = true;

    public NotesConvertOptions() { }
}

/// <summary>
/// 将模型输出的 contour / note / onset 激活矩阵转换为音符列表。
/// 移植自 Spotify basic-pitch 的 note_creation.py。
/// </summary>
public sealed class NotesConverter
{
    private readonly ModelOutput _input;

    public NotesConverter(ModelOutput input)
    {
        _input = input;
    }

    public List<Note> Convert(NotesConvertOptions opt)
    {
        var notes = ToNotesPolyphonic(opt);
        if (opt.IncludePitchBends)
            GetPitchBend(ref notes);
        return ToNoteList(notes);
    }

    private List<InterNote> ToNotesPolyphonic(NotesConvertOptions opt)
    {
        var (onsets, frames) = NotesHelper.ConstrainFrequency(_input.Onsets, _input.Notes, opt.MaxFreq, opt.MinFreq);
        if (opt.InferOnsets)
            onsets = NotesHelper.GetInferedOnsets(onsets, frames);

        var notes = new List<InterNote>();
        if (frames.Data == null) return notes;

        var remainingEnergy = new float[frames.Data.Length];
        var frameData = frames.Data;
        Array.Copy(frameData, remainingEnergy, frameData.Length);

        var onsetIdxs = NotesHelper.FindValidOnsetIndexs(onsets, opt.OnsetThreshold).Reverse();
        int frameStep = (int)frames.Shape![frames.Shape.Length - 1];
        int nFrames = frames.Shape[0];
        int nFramesMinus1 = nFrames - 1;

        foreach (var idx in onsetIdxs)
        {
            int noteStartIdx = idx / frameStep;
            int freqIdx = idx % frameStep;
            if (noteStartIdx >= nFramesMinus1) continue;

            int i = noteStartIdx + 1;
            int k = 0;
            while (i < nFrames - 1 && k < opt.EnergyThreshold)
            {
                if (remainingEnergy[i * frameStep + freqIdx] < opt.FrameThreshold) k++;
                else k = 0;
                i++;
            }
            i -= k;
            if (i - noteStartIdx <= opt.MinNoteLength) continue;

            float amplitude = 0;
            for (int j = 0; j < i - noteStartIdx; ++j)
            {
                int offset = idx + j * frameStep;
                amplitude += frameData[offset];
                remainingEnergy[offset] = 0;
                if (freqIdx < Constants.MAX_FREQ_IDX) remainingEnergy[offset + 1] = 0;
                if (freqIdx > 0) remainingEnergy[offset - 1] = 0;
            }
            amplitude /= (i - noteStartIdx);
            notes.Add(new InterNote(noteStartIdx, i, freqIdx + Constants.MIDI_OFFSET, amplitude));
        }

        if (opt.MelodiaTrick)
        {
            while (true)
            {
                int maxIdx = TensorOps.IndexOfMax(remainingEnergy);
                float maxValue = remainingEnergy[maxIdx];
                if (maxValue <= opt.FrameThreshold) break;

                int iMid = maxIdx / frameStep;
                int freqIdx = maxIdx % frameStep;
                remainingEnergy[iMid * frameStep + freqIdx] = 0;

                int i = iMid + 1;
                int k = 0;
                while (i < nFrames - 1 && k < opt.EnergyThreshold)
                {
                    int startPos = i * frameStep + freqIdx;
                    if (remainingEnergy[startPos] < opt.FrameThreshold) k++;
                    else k = 0;
                    remainingEnergy[startPos] = 0;
                    if (freqIdx < Constants.MAX_FREQ_IDX) remainingEnergy[startPos + 1] = 0;
                    if (freqIdx > 0) remainingEnergy[startPos - 1] = 0;
                    i++;
                }
                int iEnd = i - 1 - k;

                i = iMid - 1;
                k = 0;
                while (i > 0 && k < opt.EnergyThreshold)
                {
                    int startPos = i * frameStep + freqIdx;
                    if (remainingEnergy[startPos] < opt.FrameThreshold) k++;
                    else k = 0;
                    remainingEnergy[startPos] = 0;
                    if (freqIdx < Constants.MAX_FREQ_IDX) remainingEnergy[startPos + 1] = 0;
                    if (freqIdx > 0) remainingEnergy[startPos - 1] = 0;
                    i--;
                }
                int iStart = i + 1 + k;
                if (iStart < 0) throw new Exception($"iStart is: {iStart}");
                if (iEnd >= nFrames) throw new Exception($"iEnd is: {iEnd}, nFrames is: {nFrames}");

                int iLen = iEnd - iStart;
                if (iLen <= opt.MinNoteLength) continue;

                float amplitude = MathTool.Mean(frameData, iStart * frameStep + freqIdx, frameStep, iLen);
                notes.Add(new InterNote(iStart, iEnd, freqIdx + Constants.MIDI_OFFSET, amplitude));
            }
        }

        return notes;
    }

    private void GetPitchBend(ref List<InterNote> notes, int nBinsTolerance = 25)
    {
        if (_input.Contours.Data == null || notes.Count == 0) return;

        var contourSpan = _input.Contours.Data.AsSpan();
        int contourStep = (int)_input.Contours.Shape![_input.Contours.Shape.Length - 1];
        int windowLen = nBinsTolerance * 2 + 1;
        var freqGaussianSpan = NotesHelper.MakeGaussianWindow(windowLen, 5).AsSpan();

        var pitchBendSubMatrix = new float[Constants.N_FREQ_BINS_CONTOURS];
        var bends = new List<float>();

        foreach (var note in notes)
        {
            int freqIdx = (int)Math.Round(NotesHelper.MidiPitchToContourBin(note.Pitch));
            int freqStartIdx = Math.Max(freqIdx - nBinsTolerance, 0);
            int freqEndIdx = Math.Min(Constants.N_FREQ_BINS_CONTOURS, freqIdx + nBinsTolerance + 1);
            int rows = note.IEndTime - note.IStartTime;
            int cols = freqEndIdx - freqStartIdx;

            if (pitchBendSubMatrix.Length < cols)
                pitchBendSubMatrix = new float[cols];
            pitchBendSubMatrix.AsSpan().Fill(float.MinValue);

            int gaussianIdxStart = Math.Max(nBinsTolerance - freqIdx, 0);
            int gaussianIdxEnd = windowLen - Math.Max(freqIdx - (Constants.N_FREQ_BINS_CONTOURS - nBinsTolerance - 1), 0);

            bends.Clear();
            float pbShift = -(nBinsTolerance - Math.Max(0, nBinsTolerance - freqIdx));

            for (int i = 0; i < rows; ++i)
            {
                int start = (note.IStartTime + i) * contourStep + freqStartIdx;
                int mulLength = Math.Min(cols, gaussianIdxEnd - gaussianIdxStart);
                var pstart = contourSpan.Slice(start, mulLength);
                var gaussianStart = freqGaussianSpan.Slice(gaussianIdxStart, mulLength);
                TensorOps.Multiply(pstart, gaussianStart, pitchBendSubMatrix);

                int maxIdx = TensorOps.IndexOfMax(pitchBendSubMatrix.AsSpan().Slice(0, mulLength));
                bends.Add(maxIdx);
            }

            if (bends.Count > 0)
            {
                note.PitchBend = bends.ToArray();
                TensorOps.Add(note.PitchBend!, pbShift, note.PitchBend!);
            }
        }
    }

    private List<Note> ToNoteList(List<InterNote> notes)
    {
        if (notes.Count == 0 || _input.Contours.Shape == null)
            return new List<Note>();

        return notes.Select(i => new Note(
            NotesHelper.ModelFrameToTime(i.IStartTime),
            NotesHelper.ModelFrameToTime(i.IEndTime),
            i.Pitch,
            i.Amplitude,
            i.PitchBend
        )).ToList();
    }
}
