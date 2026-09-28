using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Mt3;

/// <summary>
/// MIDI writer for MT3 multi-instrument notes.
/// Notes are grouped by program (instrument) into separate tracks.
/// Drums use channel 9.
/// </summary>
public static class Mt3MidiWriter
{
    /// <summary>
    /// Write MT3 multi-instrument notes to a MIDI file.
    /// </summary>
    public static void Write(List<Mt3Note> notes, string filePath)
    {
        byte[] data = BuildBytes(notes);
        File.WriteAllBytes(filePath, data);
    }

    /// <summary>
    /// Return the MIDI file bytes.
    /// </summary>
    public static byte[] BuildBytes(List<Mt3Note> notes)
    {
        const int ticksPerQuarter = 220;
        const int bpm = 120;
        const float ticksPerSecond = ticksPerQuarter * bpm / 60f; // 440

        var events = new List<(long time, byte[] data)>();

        // Tempo meta event
        int microsecondsPerBeat = 60_000_000 / bpm;
        var tempoBytes = new byte[] { 0xFF, 0x51, 0x03,
            (byte)(microsecondsPerBeat >> 16),
            (byte)(microsecondsPerBeat >> 8),
            (byte)microsecondsPerBeat };
        events.Add((0, tempoBytes));

        // Group notes by program
        var byProgram = new Dictionary<int, List<Mt3Note>>();
        foreach (var n in notes)
        {
            int prog = n.IsDrum ? 9 : n.Program;
            if (!byProgram.TryGetValue(prog, out var list))
            {
                list = new List<Mt3Note>();
                byProgram[prog] = list;
            }
            list.Add(n);
        }

        // Program change events + notes
        int channel = 0;
        foreach (var kv in byProgram.OrderBy(k => k.Key))
        {
            int prog = kv.Key;
            var noteList = kv.Value;
            int ch = prog == 9 ? 9 : (channel++ % 9 + (channel >= 9 ? 1 : 0));
            if (ch > 15) ch = 0; // fallback

            // Program change
            byte pcStatus = (byte)(0xC0 | ch);
            events.Add((0, new byte[] { pcStatus, (byte)prog }));

            foreach (var n in noteList)
            {
                long onTick = (long)(n.StartTime * ticksPerSecond);
                long offTick = (long)(n.EndTime * ticksPerSecond);
                if (offTick <= onTick) offTick = onTick + 1;

                byte vel = (byte)Math.Clamp(n.Velocity, 1, 127);
                byte pitch = (byte)Math.Clamp(n.Pitch, 0, 127);
                byte onStatus = (byte)(0x90 | ch);
                byte offStatus = (byte)(0x80 | ch);

                events.Add((onTick, new byte[] { onStatus, pitch, vel }));
                events.Add((offTick, new byte[] { offStatus, pitch, 0 }));
            }
        }

        // Sort by time
        events.Sort((a, b) => a.time.CompareTo(b.time));

        // Build track
        var trackData = new List<byte>();
        long prevTime = 0;
        foreach (var (time, data) in events)
        {
            long delta = time - prevTime;
            if (delta < 0) delta = 0;
            WriteVariableLength(trackData, delta);
            trackData.AddRange(data);
            prevTime = time;
        }
        // End of track
        trackData.Add(0x00);
        trackData.AddRange(new byte[] { 0xFF, 0x2F, 0x00 });

        // MIDI file header
        var header = new List<byte>();
        header.AddRange(new byte[] { (byte)'M', (byte)'T', (byte)'h', (byte)'d' });
        header.AddRange(BitConverter.GetBytes(SwapEndian(6u)));
        header.AddRange(BitConverter.GetBytes(SwapEndian((ushort)0))); // format 0
        header.AddRange(BitConverter.GetBytes(SwapEndian((ushort)1))); // 1 track
        header.AddRange(BitConverter.GetBytes(SwapEndian((ushort)ticksPerQuarter)));

        // Track chunk
        var trackChunk = new List<byte>();
        trackChunk.AddRange(new byte[] { (byte)'M', (byte)'T', (byte)'r', (byte)'k' });
        trackChunk.AddRange(BitConverter.GetBytes(SwapEndian((uint)trackData.Count)));
        trackChunk.AddRange(trackData);

        var result = new byte[header.Count + trackChunk.Count];
        header.CopyTo(result, 0);
        trackChunk.CopyTo(result, header.Count);
        return result;
    }

    private static void WriteVariableLength(List<byte> buffer, long value)
    {
        long buffer2 = value & 0x7F;
        while ((value >>= 7) > 0)
        {
            buffer2 <<= 8;
            buffer2 |= (value & 0x7F) | 0x80;
        }
        while (true)
        {
            buffer.Add((byte)(buffer2 & 0xFF));
            if ((buffer2 & 0x80) != 0) buffer2 >>= 8;
            else break;
        }
    }

    private static uint SwapEndian(uint v) =>
        ((v & 0x000000FF) << 24) | ((v & 0x0000FF00) << 8) |
        ((v & 0x00FF0000) >> 8) | ((v & 0xFF000000) >> 24);

    private static ushort SwapEndian(ushort v) =>
        (ushort)(((v & 0x00FF) << 8) | ((v & 0xFF00) >> 8));
}
