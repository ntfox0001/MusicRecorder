using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BasicPitch;

/// <summary>
/// MIDI 写入参数。
/// </summary>
public record struct MidiWriteOptions
{
    public int Tempo = 120;
    public int Patch = 4;        // 电钢琴
    public bool MultiplePitchBends = false;

    public MidiWriteOptions() { }
}

/// <summary>
/// 轻量级 MIDI 文件写入器（不依赖 NAudio）。
/// 实现 basic-pitch 所需的事件：tempo、time signature、program change、
/// note on/off、pitch wheel change。
/// </summary>
public sealed class MidiWriter
{
    private readonly List<Note> _notes;

    public MidiWriter(List<Note> notes)
    {
        _notes = notes;
    }

    /// <summary>
    /// 将音符列表写入 MIDI 文件。
    /// </summary>
    public void Write(string filePath, MidiWriteOptions opt)
    {
        byte[] data = BuildBytes(opt);
        File.WriteAllBytes(filePath, data);
    }

    /// <summary>
    /// 返回 MIDI 文件的字节内容（方便 Unity 中直接保存或发送）。
    /// </summary>
    public byte[] BuildBytes(MidiWriteOptions opt)
    {
        var tempoMap = new MidiTempoMap(opt.Tempo);
        var noteList = opt.MultiplePitchBends ? _notes : DropOverlappingPitchBends();

        // 按 pitch 分轨
        var tracks = new Dictionary<int, List<MidiEvent>>();
        var orderedTracks = new List<int>();
        int channelCounter = -1;

        foreach (var note in noteList)
        {
            int trackIdx = opt.MultiplePitchBends ? note.Pitch : 0;
            List<MidiEvent> track;
            int channel;

            if (tracks.ContainsKey(trackIdx))
            {
                track = tracks[trackIdx];
                channel = track[^1].Channel;
            }
            else
            {
                track = new List<MidiEvent>();
                tracks[trackIdx] = track;
                channelCounter++;
                channel = channelCounter % 16 + 1;
                orderedTracks.Add(trackIdx);
                track.Add(new MidiEvent(0, channel, MidiEventType.ProgramChange, opt.Patch, 0));
            }

            int velocity = (int)Math.Round(127 * note.Amplitude);
            long noteonTicks = tempoMap.SecsToTicks(note.StartTime);
            long noteoffTicks = tempoMap.SecsToTicks(note.EndTime);

            track.Add(new MidiEvent(noteonTicks, channel, MidiEventType.NoteOn, note.Pitch, velocity));

            // Pitch bend
            if (note.PitchBend != null)
            {
                var pitchBendTimes = MathTool.ARange((note.StartTime, note.EndTime), note.PitchBend.Length);
                float pitchBendTicksScalar = 4096f / Constants.CONTOURS_BINS_PER_SEMITONE;
                float fNPitchBendTicks = Constants.N_PITCH_BEND_TICKS;
                float bendLimit = Constants.N_PITCH_BEND_TICKS * 2 - 1;

                var fticks = new float[note.PitchBend.Length];
                TensorOps.Multiply(note.PitchBend, pitchBendTicksScalar, fticks);
                TensorOps.Round(fticks, fticks);
                TensorOps.Add(fticks, fNPitchBendTicks, fticks);
                TensorOps.Max(fticks, 0f, fticks);
                TensorOps.Min(fticks, bendLimit, fticks);

                for (int i = 0; i < fticks.Length; i++)
                {
                    long ticks = tempoMap.SecsToTicks(pitchBendTimes[i]);
                    track.Add(new MidiEvent(ticks, channel, MidiEventType.PitchWheelChange, (int)fticks[i], 0));
                }
            }

            track.Add(new MidiEvent(noteoffTicks, channel, MidiEventType.NoteOff, note.Pitch, 0));
        }

        // 构建 MIDI 字节
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        int numTracks = orderedTracks.Count + 1; // +1 for meta track
        // Header
        bw.Write(new[] { (byte)'M', (byte)'T', (byte)'h', (byte)'d' });
        bw.Write(SwapEndian(6));          // header length
        bw.Write(SwapEndian((short)1));   // format 1
        bw.Write(SwapEndian((short)numTracks));
        bw.Write(SwapEndian((short)tempoMap.TPQ));

        // Meta track (track 0)
        var metaTrack = new List<byte>();
        // Tempo event: FF 51 03 + 3 bytes (microsec per quarter)
        int microsecPerQuarter = 60000000 / opt.Tempo;
        metaTrack.Add(0x00); // delta time
        metaTrack.Add(0xFF); metaTrack.Add(0x51); metaTrack.Add(0x03);
        metaTrack.Add((byte)(microsecPerQuarter >> 16));
        metaTrack.Add((byte)(microsecPerQuarter >> 8));
        metaTrack.Add((byte)microsecPerQuarter);
        // Time signature: FF 58 04 04 02 18 08
        metaTrack.Add(0x00);
        metaTrack.Add(0xFF); metaTrack.Add(0x58); metaTrack.Add(0x04);
        metaTrack.Add(0x04); metaTrack.Add(0x02); metaTrack.Add(0x18); metaTrack.Add(0x08);
        // End of track
        metaTrack.Add(0x00);
        metaTrack.Add(0xFF); metaTrack.Add(0x2F); metaTrack.Add(0x00);
        WriteTrack(bw, metaTrack);

        // Note tracks
        for (int i = 0; i < orderedTracks.Count; i++)
        {
            var pitch = orderedTracks[i];
            var track = tracks[pitch];
            track.Sort((a, b) =>
            {
                int t = a.AbsoluteTime.CompareTo(b.AbsoluteTime);
                return t != 0 ? t : GetEventScore(a).CompareTo(GetEventScore(b));
            });
            WriteTrack(bw, SerializeTrack(track));
        }

        return ms.ToArray();
    }

    private static int GetEventScore(MidiEvent e)
    {
        switch (e.Type)
        {
            case MidiEventType.NoteOn:
            case MidiEventType.NoteOff:
                return ((e.Channel - 1) + e.Data1) * 1000 + e.Data2;
            case MidiEventType.PitchWheelChange:
                return e.Data1;
            default:
                return 0;
        }
    }

    private static List<byte> SerializeTrack(List<MidiEvent> track)
    {
        var bytes = new List<byte>();
        long lastTime = 0;
        foreach (var e in track)
        {
            long delta = e.AbsoluteTime - lastTime;
            WriteVariableLength(bytes, delta);
            switch (e.Type)
            {
                case MidiEventType.ProgramChange:
                    bytes.Add((byte)(0xC0 | (e.Channel - 1)));
                    bytes.Add((byte)e.Data1);
                    break;
                case MidiEventType.NoteOn:
                    bytes.Add((byte)(0x90 | (e.Channel - 1)));
                    bytes.Add((byte)e.Data1);
                    bytes.Add((byte)e.Data2);
                    break;
                case MidiEventType.NoteOff:
                    bytes.Add((byte)(0x80 | (e.Channel - 1)));
                    bytes.Add((byte)e.Data1);
                    bytes.Add((byte)e.Data2);
                    break;
                case MidiEventType.PitchWheelChange:
                    bytes.Add((byte)(0xE0 | (e.Channel - 1)));
                    bytes.Add((byte)(e.Data1 & 0x7F));        // lsb
                    bytes.Add((byte)((e.Data1 >> 7) & 0x7F)); // msb
                    break;
            }
            lastTime = e.AbsoluteTime;
        }
        // End of track
        bytes.Add(0x00);
        bytes.Add(0xFF); bytes.Add(0x2F); bytes.Add(0x00);
        return bytes;
    }

    private static void WriteTrack(BinaryWriter bw, List<byte> trackData)
    {
        bw.Write(new[] { (byte)'M', (byte)'T', (byte)'r', (byte)'k' });
        bw.Write(SwapEndian(trackData.Count));
        bw.Write(trackData.ToArray());
    }

    private static void WriteVariableLength(List<byte> bytes, long value)
    {
        if (value < 0) value = 0;
        var buffer = new byte[4];
        int index = 3;
        buffer[index] = (byte)(value & 0x7F);
        while ((value >>= 7) > 0)
            buffer[--index] = (byte)((value & 0x7F) | 0x80);
        for (int i = index; i < 4; i++)
            bytes.Add(buffer[i]);
    }

    private List<Note> DropOverlappingPitchBends()
    {
        if (_notes.Count == 0) return _notes;
        var ret = _notes.OrderBy(n => n).ToList();
        for (int i = 0; i < ret.Count - 1; ++i)
        {
            var inote = ret[i];
            for (int j = i + 1; j < ret.Count; ++j)
            {
                var jnote = ret[j];
                if (jnote.StartTime >= inote.EndTime) break;
                inote.PitchBend = null;
                jnote.PitchBend = null;
            }
        }
        return ret;
    }

    private static int SwapEndian(int v) => (int)SwapEndian((uint)v);
    private static short SwapEndian(short v) => (short)SwapEndian((ushort)v);
    private static uint SwapEndian(uint v) =>
        ((v & 0x000000FF) << 24) | ((v & 0x0000FF00) << 8) |
        ((v & 0x00FF0000) >> 8) | ((v & 0xFF000000) >> 24);
    private static ushort SwapEndian(ushort v) => (ushort)(((v & 0x00FF) << 8) | ((v & 0xFF00) >> 8));

    private struct MidiTempoMap
    {
        public readonly int BPM;
        public readonly int TPQ;
        public readonly int BeatUnit;

        public MidiTempoMap(int bpm = 120, int tpq = 480, int beatUnit = 4)
        {
            BPM = bpm;
            TPQ = tpq;
            BeatUnit = beatUnit;
        }

        public long SecsToTicks(float secs)
        {
            if (secs <= 0) return 0;
            float r = secs * TPQ * BPM / (15f * BeatUnit);
            return (long)Math.Round(r);
        }
    }

    private enum MidiEventType { ProgramChange, NoteOn, NoteOff, PitchWheelChange }

    private sealed class MidiEvent
    {
        public long AbsoluteTime;
        public int Channel;
        public MidiEventType Type;
        public int Data1;
        public int Data2;

        public MidiEvent(long time, int channel, MidiEventType type, int d1, int d2)
        {
            AbsoluteTime = time;
            Channel = channel;
            Type = type;
            Data1 = d1;
            Data2 = d2;
        }
    }

}
