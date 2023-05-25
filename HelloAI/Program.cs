using TensorLib;

var trainingData = new float[,]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 0.0f }
};

float Linear(Tensor inputLayer, Tensor hiddenLayerWeights, Tensor hiddenLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    var y = Sigmoid(inputLayer * hiddenLayerWeights + hiddenLayerBias);
    y = Sigmoid(y * outputLayerWeights + outputLayerBias);

    return y[0];
}

float LossFunctionTensor(Tensor hiddenLayerWeights, Tensor hiddenLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias)
{
    var result = 0.0f;

    for (var j = 0; j < trainingData.GetLength(0); j++)
    {
        var x1 = trainingData[j, 0];
        var x2 = trainingData[j, 1];
        var testResult = trainingData[j, 2];
        
        // TODO: To Replace
        var input = new Tensor(1, 2, new float[] { x1, x2 });

        var y = Linear(input, hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias);

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

void PrintParameters(Tensor hiddenLayerWeights, Tensor hiddenLayerBias, Tensor outputLayerWeights, Tensor outputLayerBias, float loss)
{
    Console.WriteLine($"iw: {hiddenLayerWeights}");
    Console.WriteLine($"ib: {hiddenLayerBias}");
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

// TODO: Do we need to explicetely allocate the intermediate results and inputs?
// For the moment the intermediate tensors data are internally allocated.

// BUG: If we start with weights in the [-10.0f 10.0f] range we cannot train the network
var hiddenLayerWeights = new Tensor(2, 2, GenerateValue);
var hiddenLayerBias = new Tensor(1, 2, GenerateValue);
var outputLayerWeights = new Tensor(2, 1, GenerateValue);
var outputLayerBias = new Tensor(1, 1, GenerateValue);
var lossTensor = LossFunctionTensor(hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias);

Console.WriteLine("===== Tensors =====");
PrintParameters(hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias, lossTensor);

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = LossFunctionTensor(hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias);

    // HACK: Really slow!
    var copyInputWeight = hiddenLayerWeights.Copy();
    var copyInputBias = hiddenLayerBias.Copy();
    var copyOutputWeight = outputLayerWeights.Copy();
    var copyOutputBias = outputLayerBias.Copy();

    // HACK: Really slow!
    // TODO: Don't do a copy but only save the modified variable and restore it
    for (var j = 0; j < hiddenLayerWeights.ElementCount; j++)
    {
        var copy = hiddenLayerWeights.Copy();
        copy[j] += epsilon;
        var newLoss = LossFunctionTensor(copy, copyInputBias, copyOutputWeight, copyOutputBias);
        var diff = (newLoss - loss) / epsilon;
        hiddenLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < hiddenLayerBias.ElementCount; j++)
    {
        var copy = hiddenLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(copyInputWeight, copy, copyOutputWeight, copyOutputBias) - loss) / epsilon;
        hiddenLayerBias[j] -= diff * learningRate;
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
var finalLossTensor = LossFunctionTensor(hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias);
PrintParameters(hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias, finalLossTensor);

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
    
        // TODO: To Replace
        var input = new Tensor(1, 2, new float[] { x1, x2 });
        
        var y = Linear(input, hiddenLayerWeights, hiddenLayerBias, outputLayerWeights, outputLayerBias);
        Console.WriteLine($"{x1} ^ {x2} = {y}");
    }
}

