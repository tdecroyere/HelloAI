namespace TensorLib;

public record NeuralNetwork
{
    private readonly Random _random = new(28);

    public NeuralNetwork(int[] topology, Func<int, int, float>? generator = null)
    {
        ArgumentNullException.ThrowIfNull(topology);

        Weights = new Tensor[topology.Length - 1];
        Bias = new Tensor[topology.Length - 1];

        for (var i = 1; i < topology.Length; i++)
        {
            Weights.Span[i - 1] = new Tensor(topology[i - 1], topology[i], generator ?? GenerateValue);
            Bias.Span[i - 1] = new Tensor(1, topology[i], generator ?? GenerateValue);
        }
    }

    public Memory<Tensor> Weights { get; }
    public Memory<Tensor> Bias { get; }

    public Tensor Forward(Tensor inputs)
    {
        var output = inputs;

        for (var i = 0; i < Weights.Length; i++)
        {
            output = Sigmoid(output * Weights.Span[i] + Bias.Span[i]);
        }

        return output;
    }

    public override string ToString()
    {
        var stringBuilder = new StringBuilder();

        for (var i = 0; i < Weights.Length; i++)
        {
            stringBuilder.Append($"=== Layer {i} ===");
            stringBuilder.AppendLine($"{Weights.Span[i]}");
            stringBuilder.AppendLine($"{Bias.Span[i]}");
        }

        return stringBuilder.ToString();
    }
    
    private float GenerateValue(int rowIndex, int columnIndex)
    {
        return (_random.NextSingle() - 0.5f) * 2.0f;
    }
    
    private static Tensor Sigmoid(Tensor x)
    {
        // TODO: Create an exp method in factory
        var result = new Tensor(x.Rows, x.Columns);

        for (var i = 0; i < x.Rows * x.Columns; i++)
        {
            result[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
        return result;
    }
}