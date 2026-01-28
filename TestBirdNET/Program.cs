using System.Diagnostics;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace TestBirdNET;

internal static class Program
{
    private const int TARGET_SAMPLE_RATE = 48_000;
    private const int TARGET_DURATION_SECONDS = 3;
    private const int TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION_SECONDS;

    internal static void Main()
    {
        Stopwatch sw = Stopwatch.StartNew();

        string modelPath = @"C:\Users\jqntn\birding\TestBirdNET\model\birdnet\birdnet.onnx";
        string labelsPath =
            @"C:\Users\jqntn\birding\TestBirdNET\model\birdnet\label\BirdNET_GLOBAL_6K_V2.4_Labels_fr.txt";
        string mp3Path = @"C:\Users\jqntn\birding\TestBirdNET\model\birdnet\audio\blue-jay.mp3";

        string[] labels = File.ReadAllLines(labelsPath);
        Trace.Assert(labels.Length == 6522);

        float[] audio = LoadMp3(mp3Path);

        using InferenceSession session = new(modelPath);

        DenseTensor<float> inputTensor = new(audio, [1, TARGET_SAMPLES]);

        NamedOnnxValue[] inputs = [NamedOnnxValue.CreateFromTensor("input", inputTensor)];

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

        float[] scores = [.. results.First(r => r.Name == "output").AsTensor<float>()];

        IEnumerable<(int, float)> top5 = scores
            .Select((score, index) => (index, score))
            .OrderByDescending(x => x.score)
            .Take(5);

        foreach ((int index, float score) in top5)
        {
            string label = labels[index];
            string[] parts = label.Split('_', 2);

            string scientificName = parts[0];
            string commonName = parts[1];

            Console.WriteLine(
                $"{commonName} ({scientificName}): {score.ToString("F3", CultureInfo.InvariantCulture)}"
            );
        }

        sw.Stop();
        Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds} ms");
    }

    private static float[] LoadMp3(string path)
    {
        using AudioFileReader reader = new(path);

        int originalRate = reader.WaveFormat.SampleRate;
        int channels = reader.WaveFormat.Channels;
        double durationSeconds = reader.TotalTime.TotalSeconds;

        ISampleProvider sampleProvider = reader;

        if (channels > 1)
        {
            sampleProvider = new StereoToMonoSampleProvider(sampleProvider);
        }

        if (sampleProvider.WaveFormat.SampleRate != TARGET_SAMPLE_RATE)
        {
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, TARGET_SAMPLE_RATE);
        }

        float[] buffer = new float[TARGET_SAMPLES];
        float[] tmp = new float[4096];
        int index = 0;
        while (index < TARGET_SAMPLES)
        {
            int read = sampleProvider.Read(tmp, 0, tmp.Length);
            if (read == 0)
            {
                break;
            }

            int toCopy = Math.Min(read, TARGET_SAMPLES - index);
            Array.Copy(tmp, 0, buffer, index, toCopy);
            index += toCopy;
        }

        return buffer;
    }
}
