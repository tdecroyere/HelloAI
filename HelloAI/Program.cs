using TensorLib;

Tensor Linear(Tensor inputs, IAModel model)
{
    var y = Sigmoid(inputs * model.HiddenLayerWeights + model.HiddenLayerBias);
    y = Sigmoid(y * model.OutputLayerWeights + model.OutputLayerBias);

    return y;
}

float Loss(Tensor trainingInputs, Tensor trainingOutputs, IAModel model)
{
    var result = 0.0f;

    for (var i = 0; i < trainingInputs.Rows; i++)
    {
        var testResult = trainingOutputs[i, 0];
        
        var inputs = trainingInputs.RowView(i);
        var y = Linear(inputs, model);

        for (var j = 0; j < y.Columns; j++)
        {
            var d = y[0, j] - testResult;
            result += d * d;
        }
    }

    // TODO: Problem if we have multiple columns in the output tensor
    result /= trainingInputs.Rows;
    return result;
}

Tensor Sigmoid(Tensor x)
{
    // TODO: Create an exp method in factory
    var result = new Tensor(x.Rows, x.Columns);

    for (var i = 0; i < x.ElementCount; i++)
    {
        result[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
    }
    return result;
}

void TrainLayer(float loss, Tensor trainingInput, Tensor trainingOutput, Tensor layer, IAModel modelCopy, Tensor layerCopy)
{
    const float epsilon = 0.01f;
    const float learningRate = 0.1f;
    
    for (var j = 0; j < layerCopy.ElementCount; j++)
    {
        var copy = layerCopy[j];
        layerCopy[j] += epsilon;

        var newLoss = Loss(trainingInput, trainingOutput, modelCopy);
        var diff = (newLoss - loss) / epsilon;
        layerCopy[j] = copy;

        layer[j] -= diff * learningRate;
    }
}

void PrintParameters(IAModel model, float loss)
{
    Console.WriteLine($"iw: {model.HiddenLayerWeights}");
    Console.WriteLine($"ib: {model.HiddenLayerBias}");
    Console.WriteLine($"ow: {model.OutputLayerWeights}");
    Console.WriteLine($"ob: {model.OutputLayerBias}");
    Console.WriteLine($"Loss: {loss}");
}

// Test
var random = new Random(28);

float GenerateValue(int rowIndex, int columnIndex)
{
    return (random.NextSingle() - 0.5f) * 2.0f;
}

// BUG: If we start with weights in the [-10.0f 10.0f] range we cannot train the network

var model = new IAModel
{
    HiddenLayerWeights = new Tensor(2, 2, GenerateValue),
    HiddenLayerBias = new Tensor(1, 2, GenerateValue),
    OutputLayerWeights = new Tensor(2, 1, GenerateValue),
    OutputLayerBias = new Tensor(1, 1, GenerateValue)
};

var trainingData = new Tensor(4, 3, new float[]
{
    0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f
});

var trainingInput = trainingData.View(trainingData.Rows, 2);
var trainingOutput = trainingData.View(trainingData.Rows, 1, 2);

Console.WriteLine("===== Training Data =====");
Console.WriteLine(trainingInput);
Console.WriteLine(trainingOutput);

var lossTensor = Loss(trainingInput, trainingOutput, model);

Console.WriteLine("===== Tensors =====");
PrintParameters(model, lossTensor);

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = Loss(trainingInput, trainingOutput, model);
    var modelCopy = model.Copy();

    TrainLayer(loss, trainingInput, trainingOutput, model.HiddenLayerWeights, modelCopy, modelCopy.HiddenLayerWeights);
    TrainLayer(loss, trainingInput, trainingOutput, model.HiddenLayerBias, modelCopy, modelCopy.HiddenLayerBias);

    TrainLayer(loss, trainingInput, trainingOutput, model.OutputLayerWeights, modelCopy, modelCopy.OutputLayerWeights);
    TrainLayer(loss, trainingInput, trainingOutput, model.OutputLayerBias, modelCopy, modelCopy.OutputLayerBias);
}

Console.WriteLine("===== AFTER TRAINING =====");
var finalLossTensor = Loss(trainingInput, trainingOutput, model);
PrintParameters(model, finalLossTensor);

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
    
        // TODO: To Replace
        var inputs = new Tensor(1, 2, new float[] { x1, x2 });
        
        var y = Linear(inputs, model);
        Console.WriteLine($"{x1} ^ {x2} = {y}");
    }
}

public readonly record struct IAModel
{
    public required Tensor HiddenLayerWeights { get; init; }
    public required Tensor HiddenLayerBias { get; init; }
    public required Tensor OutputLayerWeights { get; init; }
    public required Tensor OutputLayerBias { get; init; }

    public IAModel Copy()
    {
        return new IAModel
        {
            HiddenLayerWeights = HiddenLayerWeights.Copy(),
            HiddenLayerBias = HiddenLayerBias.Copy(),
            OutputLayerWeights = OutputLayerWeights.Copy(),
            OutputLayerBias = OutputLayerBias.Copy()
        };
    }
}