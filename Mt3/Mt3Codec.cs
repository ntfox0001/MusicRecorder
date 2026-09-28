using System;
using System.Collections.Generic;

namespace Mt3
{
    /// <summary>
    /// MT3 event codec and note decoding state machine.
    /// Ported from mt3_infer vocab_utils.py.
    /// </summary>
    public static class Mt3Codec
    {
        // Event range definitions (matches mt3_infer vocab_utils.build_codec)
        private const int ShiftMax = 1000;     // 0..1000 (1001 values)
        private const int PitchMin = 0, PitchMax = 127;   // 128 values
        private const int VelocityMin = 0, VelocityMax = 1; // 2 values (num_velocity_bins=1)
        private const int TieMin = 0, TieMax = 0;          // 1 value
        private const int ProgramMin = 0, ProgramMax = 127; // 128 values
        private const int DrumMin = 0, DrumMax = 127;       // 128 values

        // Offsets in codec token space
        private const int ShiftOffset = 0;
        private static readonly int PitchOffset = ShiftOffset + (ShiftMax + 1);          // 1001
        private static readonly int VelocityOffset = PitchOffset + (PitchMax - PitchMin + 1); // 1129
        private static readonly int TieOffset = VelocityOffset + (VelocityMax - VelocityMin + 1); // 1131
        private static readonly int ProgramOffset = TieOffset + (TieMax - TieMin + 1);    // 1132
        private static readonly int DrumOffset = ProgramOffset + (ProgramMax - ProgramMin + 1); // 1260

        public const int NumClasses = 1388; // 1001+128+2+1+128+128
        public const int VocabSize = NumClasses + 3; // + PAD, EOS, UNK
        public const int PadToken = 0;
        public const int EosToken = 1;
        public const int UnkToken = 2;
        public const int DecoderStartToken = 0;

        private const float StepsPerSecond = 100f;
        private const float MinNoteDuration = 0.01f;
        private const int DefaultVelocity = 100;

        public enum EventType { Shift, Pitch, Velocity, Tie, Program, Drum }

        public readonly struct Event
        {
            public readonly EventType Type;
            public readonly int Value;
            public Event(EventType type, int value) { Type = type; Value = value; }
        }

        public static Event DecodeEventIndex(int index)
        {
            if (index >= ShiftOffset && index <= ShiftOffset + ShiftMax)
                return new Event(EventType.Shift, index - ShiftOffset);
            if (index >= PitchOffset && index <= PitchOffset + (PitchMax - PitchMin))
                return new Event(EventType.Pitch, PitchMin + index - PitchOffset);
            if (index >= VelocityOffset && index <= VelocityOffset + (VelocityMax - VelocityMin))
                return new Event(EventType.Velocity, VelocityMin + index - VelocityOffset);
            if (index >= TieOffset && index <= TieOffset + (TieMax - TieMin))
                return new Event(EventType.Tie, TieMin + index - TieOffset);
            if (index >= ProgramOffset && index <= ProgramOffset + (ProgramMax - ProgramMin))
                return new Event(EventType.Program, ProgramMin + index - ProgramOffset);
            if (index >= DrumOffset && index <= DrumOffset + (DrumMax - DrumMin))
                return new Event(EventType.Drum, DrumMin + index - DrumOffset);
            throw new ArgumentOutOfRangeException(nameof(index), $"Unknown event index: {index}");
        }

        private static int BinToVelocity(int velocityBin, int numVelocityBins)
        {
            if (velocityBin == 0) return 0;
            return (int)(127 * velocityBin / (double)numVelocityBins);
        }

        public class NoteDecodingState
        {
            public float CurrentTime;
            public int CurrentVelocity = DefaultVelocity;
            public int CurrentProgram;
            public readonly Dictionary<(int pitch, int program), (float onsetTime, int velocity)> ActivePitches = new();
            public readonly HashSet<(int pitch, int program)> TiedPitches = new();
            public bool IsTieSection;
            public readonly List<Mt3Note> Notes = new();
            public float TotalTime;
        }

        public static void DecodeNoteEvent(NoteDecodingState state, float time, Event evt)
        {
            if (time < state.CurrentTime)
                throw new InvalidOperationException($"event time < current time: {time} < {state.CurrentTime}");

            state.CurrentTime = time;

            switch (evt.Type)
            {
                case EventType.Pitch:
                    HandlePitch(state, evt.Value, isDrum: false);
                    break;
                case EventType.Velocity:
                    int numVelBins = VelocityMax - VelocityMin;
                    state.CurrentVelocity = BinToVelocity(evt.Value, numVelBins);
                    break;
                case EventType.Program:
                    state.CurrentProgram = evt.Value;
                    break;
                case EventType.Drum:
                    HandlePitch(state, evt.Value, isDrum: true);
                    break;
                case EventType.Tie:
                    EndTieSection(state, time);
                    break;
                case EventType.Shift:
                    break; // handled in decode_events
            }
        }

        private static void HandlePitch(NoteDecodingState state, int pitch, bool isDrum)
        {
            int program = isDrum ? 9 : state.CurrentProgram;
            var key = (pitch, program);

            if (state.IsTieSection)
            {
                if (!state.ActivePitches.ContainsKey(key))
                    throw new InvalidOperationException($"inactive pitch/program in tie section: {pitch}/{program}");
                if (state.TiedPitches.Contains(key))
                    throw new InvalidOperationException($"pitch/program already tied: {pitch}/{program}");
                state.TiedPitches.Add(key);
            }
            else if (state.CurrentVelocity == 0)
            {
                // Note off
                if (!state.ActivePitches.TryGetValue(key, out var onset))
                    throw new InvalidOperationException($"note-off for inactive pitch/program: {pitch}/{program}");
                state.ActivePitches.Remove(key);
                float endTime = MathF.Max(state.CurrentTime, onset.onsetTime + MinNoteDuration);
                state.Notes.Add(new Mt3Note(pitch, onset.onsetTime, endTime, onset.velocity, isDrum ? 0 : program, isDrum));
                state.TotalTime = MathF.Max(state.TotalTime, endTime);
            }
            else
            {
                // Note on
                if (state.ActivePitches.TryGetValue(key, out var prev))
                {
                    // Pitch already active - close previous
                    float endTime = MathF.Max(state.CurrentTime, prev.onsetTime + MinNoteDuration);
                    state.Notes.Add(new Mt3Note(pitch, prev.onsetTime, endTime, prev.velocity, isDrum ? 0 : program, isDrum));
                    state.TotalTime = MathF.Max(state.TotalTime, endTime);
                }
                state.ActivePitches[key] = (state.CurrentTime, state.CurrentVelocity);
            }
        }

        private static void EndTieSection(NoteDecodingState state, float time)
        {
            if (!state.IsTieSection)
                throw new InvalidOperationException("tie section end event when not in tie section");

            var toClose = new List<(int pitch, int program)>();
            foreach (var kv in state.ActivePitches)
            {
                if (state.TiedPitches.Contains(kv.Key)) continue;
                toClose.Add(kv.Key);
            }

            foreach (var key in toClose)
            {
                var onset = state.ActivePitches[key];
                bool isDrum = key.program == 9;
                float endTime = MathF.Max(time, onset.onsetTime + MinNoteDuration);
                state.Notes.Add(new Mt3Note(key.pitch, onset.onsetTime, endTime, onset.velocity, isDrum ? 0 : key.program, isDrum));
                state.TotalTime = MathF.Max(state.TotalTime, endTime);
                state.ActivePitches.Remove(key);
            }

            state.TiedPitches.Clear();
            state.IsTieSection = false;
        }

        /// <summary>
        /// Decode a token sequence (codec indices, i.e. raw token - 3) for one segment.
        /// </summary>
        public static void DecodeSegment(NoteDecodingState state, int[] tokens, float startTime, float? maxTime)
        {
            state.IsTieSection = true;
            state.TiedPitches.Clear();

            int curSteps = 0;
            float curTime = startTime;
            for (int i = 0; i < tokens.Length; i++)
            {
                int token = tokens[i];
                Event evt;
                try { evt = DecodeEventIndex(token); }
                catch { continue; }

                if (evt.Type == EventType.Shift)
                {
                    curSteps += evt.Value;
                    curTime = startTime + curSteps / StepsPerSecond;
                    if (maxTime.HasValue && curTime > maxTime.Value)
                        break;
                }
                else
                {
                    // curTime persists from last shift; only curSteps resets
                    curSteps = 0;
                    try
                    {
                        DecodeNoteEvent(state, curTime, evt);
                    }
                    catch { continue; }
                }
            }

            state.IsTieSection = false;
        }

        /// <summary>
        /// Flush remaining active notes and return the note list.
        /// </summary>
        public static List<Mt3Note> Flush(NoteDecodingState state)
        {
            foreach (var kv in state.ActivePitches)
            {
                bool isDrum = kv.Key.program == 9;
                float endTime = MathF.Max(state.CurrentTime, kv.Value.onsetTime + MinNoteDuration);
                state.Notes.Add(new Mt3Note(kv.Key.pitch, kv.Value.onsetTime, endTime, kv.Value.velocity, isDrum ? 0 : kv.Key.program, isDrum));
                state.TotalTime = MathF.Max(state.TotalTime, endTime);
            }
            state.ActivePitches.Clear();
            return state.Notes;
        }
    }

    public readonly struct Mt3Note
    {
        public readonly int Pitch;
        public readonly float StartTime;
        public readonly float EndTime;
        public readonly int Velocity;
        public readonly int Program;
        public readonly bool IsDrum;

        public Mt3Note(int pitch, float startTime, float endTime, int velocity, int program, bool isDrum)
        {
            Pitch = pitch;
            StartTime = startTime;
            EndTime = endTime;
            Velocity = velocity;
            Program = program;
            IsDrum = isDrum;
        }
    }
}
