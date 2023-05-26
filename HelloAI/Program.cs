using TensorLib;
using HelloIA;

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
        var inputs = trainingInputs.ViewRow(i);
        var outputs = trainingOutputs.ViewRow(i);

        var y = Linear(inputs, model);

        for (var j = 0; j < y.Columns; j++)
        {
            var d = y[j] - outputs[j];
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

    for (var i = 0; i < x.Rows * x.Columns; i++)
    {
        result[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
    }
    return result;
}

void CalculateGradients(float loss, Tensor trainingInput, Tensor trainingOutput, IAModel model, Tensor layer, Tensor layerGradients)
{
    const float epsilon = 0.01f;
    
    for (var j = 0; j < layer.Rows * layer.Columns; j++)
    {
        var copy = layer[j];
        layer[j] += epsilon;

        var newLoss = Loss(trainingInput, trainingOutput, model);
        var diff = (newLoss - loss) / epsilon;

        layer[j] = copy;
        layerGradients[j] = diff;
    }
}

void Learn(Tensor layer, Tensor layerGradients)
{
    const float learningRate = 0.1f;
    
    for (var j = 0; j < layer.Rows * layer.Columns; j++)
    {
        layer[j] -= layerGradients[j] * learningRate;
    }
}

void PrintParameters(IAModel model, float loss)
{
    Console.WriteLine($"iw: {model.HiddenLayerWeights}");
    Console.WriteLine($"ib: {model.HiddenLayerBias}");
    Console.WriteLine($"ow: {model.OutputLayerWeights}");
    Console.WriteLine($"ob: {model.OutputLayerBias}");

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"Loss: {loss}");
    Console.ForegroundColor = ConsoleColor.Gray;
}

// BUG: If we start with weights in the [-10.0f 10.0f] range we cannot train the network
var model = new IAModel(withRandomValues: true);
var gradients = new IAModel(withRandomValues: false);

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
for (var i = 0; i < 100_000; i++)
{
    var loss = Loss(trainingInput, trainingOutput, model);

    CalculateGradients(loss, trainingInput, trainingOutput, model, model.HiddenLayerWeights, gradients.HiddenLayerWeights);
    CalculateGradients(loss, trainingInput, trainingOutput, model, model.HiddenLayerBias, gradients.HiddenLayerBias);
    CalculateGradients(loss, trainingInput, trainingOutput, model, model.OutputLayerWeights, gradients.OutputLayerWeights);
    CalculateGradients(loss, trainingInput, trainingOutput, model, model.OutputLayerBias, gradients.OutputLayerBias);

    Learn(model.HiddenLayerWeights, gradients.HiddenLayerWeights);
    Learn(model.HiddenLayerBias, gradients.HiddenLayerBias);
    Learn(model.OutputLayerWeights, gradients.OutputLayerWeights);
    Learn(model.OutputLayerBias, gradients.OutputLayerBias);
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

namespace HelloIA
{
    public record IAModel
    {
        private readonly Random _random = new(28);

        public IAModel(bool withRandomValues)
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
}