using Microsoft.ML.Data;

namespace MlDojoBackend;

public class ModelInput
{
    [LoadColumn(0)] // Maps to the first column in the CSV
    public string Text { get; set; } = string.Empty;

    [LoadColumn(1)] // Maps to the second column in the CSV
    public bool Label { get; set; }
}