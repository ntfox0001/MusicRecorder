using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace Mp3ToSheet;

public partial class MainWindow : Window
{
    private EngineRunner? _runner;
    private bool _running;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += (_, _) =>
        {
            _runner = new EngineRunner(AppendLog);
            try
            {
                _runner.Preload(EngineKind.BasicPitch);
                AppendLog("[就绪] Basic Pitch 模型已加载（MT3 / Madmom 首次使用时再加载）");
            }
            catch (Exception ex)
            {
                AppendLog("[错误] 模型加载失败: " + ex.Message);
            }
        };
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
        if (_running || _runner == null) return;

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

        var engine = SelectedEngine();

        _running = true;
        BtnConvert.IsEnabled = false;
        BtnBrowse.IsEnabled = false;
        TxtStatus.Text = "转换中...";
        TxtLog.Clear();
        TxtMidiOut.Text = "";

        try
        {
            AppendLog($"===== {engine} =====");
            var (outPath, summary) = await Task.Run(() => _runner.Run(input, threshold, engine));
            TxtStatus.Text = "完成";
            TxtMidiOut.Text = outPath;
            AppendLog(summary);
            AppendLog($"输出: {outPath}");
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

    private EngineKind SelectedEngine()
    {
        string tag = (CmbEngine.SelectedItem as ComboBoxItem)?.Tag as string ?? "basicpitch";
        return EngineRunner.Parse(tag);
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
        _runner?.Dispose();
        base.OnClosed(e);
    }
}
