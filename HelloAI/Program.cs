using TensorLib;

float Loss(Tensor trainingInputs, Tensor trainingOutputs, NeuralNetwork model)
{
    var result = 0.0f;

    for (var i = 0; i < trainingInputs.Rows; i++)
    {
        var inputs = trainingInputs.ViewRow(i);
        var outputs = trainingOutputs.ViewRow(i);

        var y = model.Forward(inputs);

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

void CalculateGradients(float loss, Tensor trainingInput, Tensor trainingOutput, NeuralNetwork model, NeuralNetwork gradients)
{
    for (var i = 0; i < model.Weights.Length; i++)
    {
        var layer = model.Weights.Span[i];
        var gradientLayer = gradients.Weights.Span[i];

        CalculateGradientsLayer(loss, trainingInput, trainingOutput, model, layer, gradientLayer);
    }
    
    for (var i = 0; i < model.Bias.Length; i++)
    {
        var layer = model.Bias.Span[i];
        var gradientLayer = gradients.Bias.Span[i];

        CalculateGradientsLayer(loss, trainingInput, trainingOutput, model, layer, gradientLayer);
    }
}

void CalculateGradientsLayer(float loss, Tensor trainingInput, Tensor trainingOutput, NeuralNetwork model, Tensor layer, Tensor layerGradients)
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

void Learn(NeuralNetwork model, NeuralNetwork gradients)
{
    const float learningRate = 0.1f;
    
    for (var i = 0; i < model.Weights.Length; i++)
    {
        var layer = model.Weights.Span[i];
        var layerGradients = gradients.Weights.Span[i];

        for (var j = 0; j < layer.Rows * layer.Columns; j++)
        {
            layer[j] -= layerGradients[j] * learningRate;
        }
    }
    
    for (var i = 0; i < model.Bias.Length; i++)
    {
        var layer = model.Bias.Span[i];
        var layerGradients = gradients.Bias.Span[i];

        for (var j = 0; j < layer.Rows * layer.Columns; j++)
        {
            layer[j] -= layerGradients[j] * learningRate;
        }
    }
}

var trainingData = new Tensor(4, 3, new float[]
{
    0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f
});

var trainingInput = trainingData.View(trainingData.Rows, 2);
var trainingOutput = trainingData.View(trainingData.Rows, 1, 2);

// BUG: If we start with weights in the [-10.0f 10.0f] range we cannot train the network
var topology = new int[] { 2, 2, 1 };

var model = new NeuralNetwork(topology);
var gradients = new NeuralNetwork(topology);

Console.WriteLine("===== Training Data =====");
Console.WriteLine(trainingInput);
Console.WriteLine(trainingOutput);

var loss = Loss(trainingInput, trainingOutput, model);

Console.WriteLine("===== Tensors =====");
Console.WriteLine(model);
Console.WriteLine($"Loss: {loss}");

// Training
for (var i = 0; i < 100_000; i++)
{
    loss = Loss(trainingInput, trainingOutput, model);

    CalculateGradients(loss, trainingInput, trainingOutput, model, gradients);
    Learn(model, gradients);
}

loss = Loss(trainingInput, trainingOutput, model);

Console.WriteLine("===== AFTER TRAINING =====");
Console.WriteLine(model);
Console.WriteLine($"Loss: {loss}");

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
    
        // TODO: To Replace
        var inputs = new Tensor(1, 2, new float[] { x1, x2 });
        
        var y = model.Forward(inputs);
        Console.WriteLine($"{x1} ^ {x2} = {y}");
    }
}