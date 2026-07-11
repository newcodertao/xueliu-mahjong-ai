using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class XueliuRealtimeLauncher
{
    [STAThread]
    private static void Main()
    {
        try
        {
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string projectRoot = FindProjectRoot(exeDir);
            string pythonw = Path.Combine(projectRoot, ".venv", "Scripts", "pythonw.exe");
            string python = Path.Combine(projectRoot, ".venv", "Scripts", "python.exe");
            string interpreter = File.Exists(pythonw) ? pythonw : python;

            if (!File.Exists(interpreter))
            {
                MessageBox.Show(
                    "Python virtual environment was not found. Please check .venv under project root.\\n\\n" + projectRoot,
                    "Xueliu Mahjong Assistant",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = interpreter,
                Arguments = "-m xueliu_ai realtime-ui",
                WorkingDirectory = projectRoot,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            startInfo.EnvironmentVariables["PYTHONPATH"] = Path.Combine(projectRoot, "src");
            Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "Xueliu Mahjong Assistant failed to start",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }

    private static string FindProjectRoot(string start)
    {
        DirectoryInfo dir = new DirectoryInfo(start);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml"))
                && Directory.Exists(Path.Combine(dir.FullName, "src")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return Path.GetFullPath(Path.Combine(start, ".."));
    }
}
