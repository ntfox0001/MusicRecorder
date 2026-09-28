using System;

namespace BasicPitch;

/// <summary>
/// 最终输出的音符。
/// </summary>
public sealed class Note : IComparable<Note>
{
    public readonly float StartTime;
    public readonly float EndTime;
    public readonly int Pitch;
    public readonly float Amplitude;
    public float[]? PitchBend;

    public Note(float startTime, float endTime, int pitch, float amplitude, float[]? pitchBend)
    {
        StartTime = startTime;
        EndTime = endTime;
        Pitch = pitch;
        Amplitude = amplitude;
        PitchBend = pitchBend;
    }

    public int CompareTo(Note? other)
    {
        if (other == null) return 1;
        float fcmp = StartTime - other.StartTime;
        if (fcmp != 0f) return Math.Sign(fcmp);
        fcmp = EndTime - other.EndTime;
        if (fcmp != 0f) return Math.Sign(fcmp);
        int icmp = Pitch - other.Pitch;
        if (icmp != 0) return Math.Sign(icmp);
        fcmp = Amplitude - other.Amplitude;
        if (fcmp != 0f) return Math.Sign(fcmp);
        int l = PitchBend == null ? -1 : PitchBend.Length;
        int r = other.PitchBend == null ? -1 : other.PitchBend.Length;
        return Math.Sign(l - r);
    }
}

/// <summary>
/// 后处理中间态音符（帧索引而非秒）。
/// </summary>
internal sealed class InterNote
{
    public int IStartTime;
    public int IEndTime;
    public int Pitch;
    public float Amplitude;
    public float[]? PitchBend;

    public InterNote(int start, int end, int pitch, float amplitude)
    {
        IStartTime = start;
        IEndTime = end;
        Pitch = pitch;
        Amplitude = amplitude;
    }
}
