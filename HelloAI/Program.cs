using TensorLib;

var trainingData = new float[,]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 0.0f }
};

float Linear(float x1, float x2, Tensor inputLayerWeights, Tensor inputLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    var input = new Tensor(1, 2, new float[] { x1, x2 });

    var y = Sigmoid(input * inputLayerWeights + inputLayerBias);
    y = Sigmoid(y * outputLayerWeights + outputLayerBias);

    return y[0];
}

float LossFunctionTensor(Tensor inputLayerWeights, Tensor inputLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    var result = 0.0f;

    for (var j = 0; j < trainingData.GetLength(0); j++)
    {
        var x1 = trainingData[j, 0];
        var x2 = trainingData[j, 1];
        var testResult = trainingData[j, 2];

        var y = Linear(x1, x2, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);

        var d = y - testResult;
        result += d * d;
    }

    result /= trainingData.GetLength(0);
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

void PrintParameters(Tensor inputLayerWeights, Tensor inputLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias, float loss)
{
    Console.WriteLine($"iw: {inputLayerWeights}");
    Console.WriteLine($"ib: {inputLayerBias}");
    Console.WriteLine($"ow: {outputLayerWeights}");
    Console.WriteLine($"ob: {outputLayerBias}");
    Console.WriteLine($"Loss: {loss}");
}

var epsilon = 0.01f;
var learningRate = 0.1f;

// Test
var random = new Random(28);

float GenerateValue(int rowIndex, int columnIndex)
{
    return (random.NextSingle() - 0.5f) * 2.0f;
}

// BUG: If we start with weights in the [-10.0f 10.0f] range we cannot train the network
var inputLayerWeights = new Tensor(2, 2, GenerateValue);
var inputLayerBias = new Tensor(1, 2, GenerateValue);
var outputLayerWeights = new Tensor(2, 1, GenerateValue);
var outputLayerBias = new Tensor(1, 1, GenerateValue);
var lossTensor = LossFunctionTensor(inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);

Console.WriteLine("===== Tensors =====");
PrintParameters(inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias, lossTensor);

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = LossFunctionTensor(inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);

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
        var newLoss = LossFunctionTensor(copy, copyInputBias, copyOutputWeight, copyOutputBias);
        var diff = (newLoss - loss) / epsilon;
        inputLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < inputLayerBias.ElementCount; j++)
    {
        var copy = inputLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(copyInputWeight, copy, copyOutputWeight, copyOutputBias) - loss) / epsilon;
        inputLayerBias[j] -= diff * learningRate;
    }

    for (var j = 0; j < outputLayerWeights.ElementCount; j++)
    {
        var copy = outputLayerWeights.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(copyInputWeight, copyInputBias, copy, copyOutputBias) - loss) / epsilon;
        outputLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < outputLayerBias.ElementCount; j++)
    {
        var copy = outputLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(copyInputWeight, copyInputBias, copyOutputWeight, copy) - loss) / epsilon;
        outputLayerBias[j] -= diff * learningRate;
    }

    //Console.WriteLine($"w: {outputLayerWeights} b: {outputLayerBias}, Loss: {loss}");
}

Console.WriteLine("===== AFTER TRAINING =====");
var finalLossTensor = LossFunctionTensor(inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);
PrintParameters(inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias, finalLossTensor);

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
        
        var y = Linear(x1, x2, inputLayerWeights, inputLayerBias, outputLayerWeights, outputLayerBias);
        Console.WriteLine($"{x1} ^ {x2} = {y}");
    }
}

