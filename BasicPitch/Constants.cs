namespace BasicPitch;

/// <summary>
/// Basic Pitch 模型常量（与 Spotify basic-pitch 及 basic-pitch-dotnet 一致）。
/// </summary>
internal static class Constants
{
    public const int FFT_HOP = 256;
    public const int N_OVERLAPPING_FRAMES = 30;
    public const int OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP;
    public const int AUDIO_SAMPLE_RATE = 22050;
    public const int AUDIO_WINDOW_LEN = 2;
    public const int AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LEN - FFT_HOP;
    public const int HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN;
    public const int ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE / FFT_HOP;
    public const int MIDI_OFFSET = 21;
    public const int MAX_FREQ_IDX = 87;
    public const int ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LEN;
    public const int CONTOURS_BINS_PER_SEMITONE = 3;
    public const int ANNOTATIONS_N_SEMITONES = 88;
    public const float ANNOTATIONS_BASE_FREQUENCY = 27.5f;
    public const int N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;
    public const int N_PITCH_BEND_TICKS = 8192;
}
