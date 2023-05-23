using TensorLib;

var trainingData = new float[,]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 0.0f }
};

float Linear(Torch torch, float x1, float x2, Tensor inputLayerWeights, Tensor inputLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    // TODO: Do some kind of tensor repeating when we are out of bound?
    var input = torch.Tensor(new float[] { x1, x2 }, 2, 1);

    var y = Sigmoid(torch, input * inputLayerWeights + inputLayerBias);
    y = Sigmoid(torch, y * outputLayerWeights + outputLayerBias);

    //y = Sigmoid(torch, input * outputLayerWeights + outputLayerBias);

    return y[0];
}

float LossFunctionTensor(Torch torch, Tensor inputLayerWeights, Tensor inputLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    var result = 0.0f;

    for (var j = 0; j < trainingData.GetLength(0); j++)
    {
        var x1 = trainingData[j, 0];
        var x2 = trainingData[j, 1];
        var testResult = trainingData[j, 2];

        var y = Linear(torch, x1, x2, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);

        var d = y - testResult;
        result += d * d;
    }

    result /= trainingData.GetLength(0);
    return result;
}

Tensor Sigmoid(Torch torch, Tensor x)
{
    // TODO: Create an exp method in torch
    var result = torch.Zeros((int)x.Size.X, (int)x.Size.Y);

    for (var i = 0; i < x.ElementCount; i++)
    {
        result[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
    }
    return result;
}

var epsilon = 0.01f;
var learningRate = 0.1f;

// Test
var torch = new Torch();

// BUG: If we multiply the weights there is an error :()
var inputLayerWeights = (torch.Random(2, 2) - 0.5f) * 2.0f;
var inputLayerBias = (torch.Random(2, 1) - 0.5f) * 2.0f;
var outputLayerWeights = (torch.Random(1, 2) - 0.5f) * 2.0f;
var outputLayerBias = (torch.Random(1, 1) - 0.5f) * 2.0f;

Console.WriteLine("===== Tensors =====");

var lossTensor = LossFunctionTensor(torch, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);
Console.WriteLine($"iw: {inputLayerWeights} ib: {inputLayerBias}, ow: {outputLayerWeights} ob: {outputLayerBias}, Loss: {lossTensor}");

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = LossFunctionTensor(torch, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);

    // HACK: Really slow!
    var copyInputWeight = inputLayerWeights.Copy();
    var copyInputBias = inputLayerBias.Copy();
    var copyOutputWeight = outputLayerWeights.Copy();
    var copyOutputBias = outputLayerBias.Copy();

    // HACK: Really slow!
    // TODO: Don't do a copy but only save the modified variable and restore it
    for (var j = 0; j < inputLayerWeights.ElementCount; j++)
    {
        var copy = inputLayerWeights.Copy();
        copy[j] += epsilon;
        var newLoss = LossFunctionTensor(torch, copy, copyInputBias, copyOutputWeight, copyOutputBias);
        var diff = (newLoss - loss) / epsilon;
        inputLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < inputLayerBias.ElementCount; j++)
    {
        var copy = inputLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(torch, copyInputWeight, copy, copyOutputWeight, copyOutputBias) - loss) / epsilon;
        inputLayerBias[j] -= diff * learningRate;
    }

    for (var j = 0; j < outputLayerWeights.ElementCount; j++)
    {
        var copy = outputLayerWeights.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(torch, copyInputWeight, copyInputBias, copy, copyOutputBias) - loss) / epsilon;
        outputLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < outputLayerBias.ElementCount; j++)
    {
        var copy = outputLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(torch, copyInputWeight, copyInputBias, copyOutputWeight, copy) - loss) / epsilon;
        outputLayerBias[j] -= diff * learningRate;
    }

    //Console.WriteLine($"w: {outputLayerWeights} b: {outputLayerBias}, Loss: {loss}");
}

var finalLossTensor = LossFunctionTensor(torch, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);
Console.WriteLine($"iw: {inputLayerWeights} ib: {inputLayerBias}, ow: {outputLayerWeights} ob: {outputLayerBias}, Loss: {finalLossTensor}");

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
        
        var y = Linear(torch, x1, x2, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);
        Console.WriteLine($"{x1} ^ {x2} = {y}");
    }
}

