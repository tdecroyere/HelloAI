namespace TensorLib;

public record NeuralNetwork
{
    private readonly Random _random = new(28);

    public NeuralNetwork(int[] topology, Func<int, int, float>? generator = null)
    {
        ArgumentNullException.ThrowIfNull(topology);

        // TODO: Can we allocate one Tensor that contains the data and then
        // Separate tensors that acts as views?

        Weights = new Tensor[topology.Length - 1];
        Bias = new Tensor[topology.Length - 1];
        Activations = new Tensor[topology.Length];

        Activations.Span[0] = new Tensor(1, topology[0], generator ?? GenerateValue);

        for (var i = 1; i < topology.Length; i++)
        {
            Weights.Span[i - 1] = new Tensor(topology[i - 1], topology[i], generator ?? GenerateValue);
            Bias.Span[i - 1] = new Tensor(1, topology[i], generator ?? GenerateValue);
            Activations.Span[i] = new Tensor(1, topology[i], generator ?? GenerateValue);
        }
    }

    public Memory<Tensor> Weights { get; }
    public Memory<Tensor> Bias { get; }

    // TODO: Do we need to store the activations here, it is used temporary between
    // forward and back propagation because we need the intermediate results of each layers
    public Memory<Tensor> Activations { get; }

    public Tensor Forward(Tensor inputs)
    {
        if (Activations.Span[0].Columns != inputs.Columns)
        {
            throw new ArgumentException(nameof(inputs), "Columns of inputs should have the correct number.");
        }

        Activations.Span[0] = inputs.Copy();

        for (var i = 0; i < Weights.Length; i++)
        {
            Activations.Span[i + 1] = Sigmoid(Activations.Span[i] * Weights.Span[i] + Bias.Span[i]);
        }

        return Activations.Span[^1];
    }

    public void Zero()
    {
        for (var i = 0; i < Weights.Length; i++)
        {
            Weights.Span[i].Zero();
        }

        for (var i = 0; i < Bias.Length; i++)
        {
            Bias.Span[i].Zero();
        }
        
        for (var i = 0; i < Activations.Length; i++)
        {
            Activations.Span[i].Zero();
        }
    }

    public override string ToString()
    {
        var stringBuilder = new StringBuilder();

        for (var i = 0; i < Weights.Length; i++)
        {
            stringBuilder.Append($"=== Layer {i} ===");
            stringBuilder.AppendLine($"{Weights.Span[i].ToString("Weights")}");
            stringBuilder.AppendLine($"{Bias.Span[i].ToString("Bias")}");
            stringBuilder.AppendLine($"{Activations.Span[i].ToString("Activations")}");
        }
        
        stringBuilder.AppendLine($"=== Output ===");
        stringBuilder.AppendLine($"{Activations.Span[^1].ToString("Activations")}");

        return stringBuilder.ToString();
    }
    
    private float GenerateValue(int rowIndex, int columnIndex)
    {
        // BUG: Again here 0.0 to 1.0 seems to work better :(
        return _random.NextSingle();
        //return (_random.NextSingle() - 0.5f) * 2.0f;
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
