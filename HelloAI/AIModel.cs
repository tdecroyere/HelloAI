using TensorLib;

namespace HelloIA;
    
public record AIModel
{
    private readonly Random _random = new(28);

    public AIModel(bool withRandomValues)
    {
        if (withRandomValues)
        {
            HiddenLayerWeights = new Tensor(2, 2, GenerateValue);
            HiddenLayerBias = new Tensor(1, 2, GenerateValue);
            OutputLayerWeights = new Tensor(2, 1, GenerateValue);
            OutputLayerBias = new Tensor(1, 1, GenerateValue);
        }
        else
        {
            HiddenLayerWeights = new Tensor(2, 2);
            HiddenLayerBias = new Tensor(1, 2);
            OutputLayerWeights = new Tensor(2, 1);
            OutputLayerBias = new Tensor(1, 1);
        }
    }

    public Tensor HiddenLayerWeights { get; init; }
    public Tensor HiddenLayerBias { get; init; }
    public Tensor OutputLayerWeights { get; init; }
    public Tensor OutputLayerBias { get; init; }

    private float GenerateValue(int rowIndex, int columnIndex)
    {
        return (_random.NextSingle() - 0.5f) * 2.0f;
    }
}