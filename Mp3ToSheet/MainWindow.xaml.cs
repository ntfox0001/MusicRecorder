using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using BasicPitch;
using Microsoft.Win32;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Mp3ToSheet;

public partial class MainWindow : Window
{
    private BasicPitchConverter? _converter;
    private bool _running;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += (_, _) => InitModel();
    }

    private void InitModel()
    {
        try
        {
            _converter = new BasicPitchConverter();
            AppendLog("[就绪] Basic Pitch 模型已加载");
        }
        catch (Exception ex)
        {
            AppendLog("[错误] 模型加载失败: " + ex.Message);
        }
    }

    private void BtnBrowse_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "音频文件 (*.mp3;*.wav;*.flac)|*.mp3;*.wav;*.flac|所有文件 (*.*)|*.*",
            Title = "选择音频文件"
        };
        if (dlg.ShowDialog() == true)
            TxtInput.Text = dlg.FileName;
    }

    private async void BtnConvert_Click(object sender, RoutedEventArgs e)
    {
        if (_running || _converter == null) return;

        var input = TxtInput.Text.Trim();
        if (string.IsNullOrEmpty(input) || !File.Exists(input))
        {
            MessageBox.Show("请先选择一个有效的音频文件。", "提示",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!float.TryParse(TxtThreshold.Text, out var threshold))
        {
            MessageBox.Show("帧激活阈值格式不正确。", "提示",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        _running = true;
        BtnConvert.IsEnabled = false;
        BtnBrowse.IsEnabled = false;
        TxtStatus.Text = "转换中...";
        TxtLog.Clear();
        TxtMidiOut.Text = "";

        try
        {
            var (midiPath, noteCount) = await Task.Run(() => RunConversion(input, threshold));
            TxtStatus.Text = "完成";
            TxtMidiOut.Text = midiPath;
            AppendLog($"检测到 {noteCount} 个音符，MIDI 已保存: {midiPath}");
        }
        catch (Exception ex)
        {
            AppendLog("[异常] " + ex.Message);
            TxtStatus.Text = "失败";
        }
        finally
        {
            _running = false;
            BtnConvert.IsEnabled = true;
            BtnBrowse.IsEnabled = true;
        }
    }

    /// <summary>
    /// 用 NAudio 加载音频并重采样为 22050Hz 单声道 float[]，
    /// 然后调用 BasicPitchConverter 生成 MIDI。
    /// </summary>
    private (string midiPath, int noteCount) RunConversion(string input, float threshold)
    {
        AppendLog($"[1/3] 加载音频: {input}");
        float[] audio;
        int sampleRate;
        using (var reader = new AudioFileReader(input))
        {
            sampleRate = reader.WaveFormat.SampleRate;
            // 用 WDL 重采样器转到 22050Hz（比 MediaFoundationResampler 更稳定）
            var resampler = new WdlResamplingSampleProvider(reader, 22050);
            int channels = resampler.WaveFormat.Channels;
            int sr = resampler.WaveFormat.SampleRate;

            using var ms = new MemoryStream();
            var floatBuffer = new float[sr * channels * 2];
            int read;
            while ((read = resampler.Read(floatBuffer, 0, floatBuffer.Length)) > 0)
            {
                for (int i = 0; i < read; i++)
                {
                    byte[] bytes = BitConverter.GetBytes(floatBuffer[i]);
                    ms.Write(bytes, 0, 4);
                }
            }

            var byteData = ms.ToArray();
            int sampleCount = byteData.Length / 4;
            var interleaved = new float[sampleCount];
            Buffer.BlockCopy(byteData, 0, interleaved, 0, byteData.Length);

            // 混合为单声道
            if (channels == 1)
                audio = interleaved;
            else
            {
                int n = sampleCount / channels;
                audio = new float[n];
                for (int i = 0; i < n; i++)
                {
                    float sum = 0;
                    for (int c = 0; c < channels; c++)
                        sum += interleaved[i * channels + c];
                    audio[i] = sum / channels;
                }
            }
        }
        AppendLog($"      采样率={sampleRate}Hz -> 22050Hz, 时长={audio.Length / 22050f:F2}s");

        AppendLog("[2/3] Basic Pitch 音高检测中...");
        var options = new NotesConvertOptions
        {
            FrameThreshold = threshold,
            OnsetThreshold = 0.5f,
        };
        var notes = _converter!.Convert(audio, 22050, options);
        AppendLog($"      检测到 {notes.Count} 个音符");

        AppendLog("[3/3] 生成 MIDI...");
        string baseName = Path.Combine(Path.GetDirectoryName(input)!,
            Path.GetFileNameWithoutExtension(input));
        string midiPath = baseName + ".mid";
        _converter.ConvertToMidi(audio, 22050, midiPath, options);
        AppendLog($"      MIDI 已保存: {midiPath}");
        return (midiPath, notes.Count);
    }

    private void AppendLog(string line)
    {
        Dispatcher.Invoke(() =>
        {
            TxtLog.AppendText(line + Environment.NewLine);
            TxtLog.ScrollToEnd();
        });
    }

    protected override void OnClosed(EventArgs e)
    {
        _converter?.Dispose();
        base.OnClosed(e);
    }
}
