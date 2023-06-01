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

    result /= trainingInputs.Rows;
    return result;
}

void CalculateGradients(Tensor trainingInput, Tensor trainingOutput, NeuralNetwork model, NeuralNetwork gradients)
{
    gradients.Zero();

    var loss = Loss(trainingInput, trainingOutput, model);

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

float BackwardActivation(float x)
{
    return x > 0.0f ? 1.0f : 0.0f; // RELU
    //return 1.0f - x * x; // TanH
    //return x * (1 - x); // Sigmoid
}

void CalculateBackPropagation(Tensor trainingInput, Tensor trainingOutput, NeuralNetwork model, NeuralNetwork gradients)
{
    gradients.Zero();
    var trainingCount = trainingInput.Rows;
    
    for (var i = 0; i < trainingCount; i++)
    {
        var inputs = trainingInput.ViewRow(i);
        var outputs = trainingOutput.ViewRow(i);
        var y = model.Forward(inputs);
    
        // IMPORTANT: this one is critical!!!!
        for (var j = 0; j < gradients.Activations.Length; j++)
        {
            gradients.Activations.Span[j].Zero();
        }

        // Assign result gradient to the output gradient activation layer
        var outputGradientActivation = gradients.Activations.Span[^1];

        for (var j = 0; j < y.Columns; j++)
        {
            outputGradientActivation[j] = y[j] - outputs[j]; 
        }

        // Process current layer
        for (var l = model.Weights.Length; l > 0; l--)
        {
            // TODO: This could be done with tensor operations
            for (var j = 0; j < model.Activations.Span[l].Columns; j++)
            {
                var a = model.Activations.Span[l][0, j];
                var da = gradients.Activations.Span[l][0, j];

                gradients.Bias.Span[l - 1][0, j] += 2 * da * BackwardActivation(a);

                for (var k = 0; k < model.Activations.Span[l - 1].Columns; k++)
                {
                    // pa = previous activation
                    var pa = model.Activations.Span[l - 1][0, k];
                    var w = model.Weights.Span[l - 1][k, j];

                    gradients.Weights.Span[l - 1][k, j] += 2 * da * BackwardActivation(a) * pa;
                    gradients.Activations.Span[l - 1][0, k] += 2 * da * BackwardActivation(a) * w;
                }
            }
        }
    }

    for (var i = 0; i < gradients.Weights.Length; i++)
    {
        for (var j = 0; j < gradients.Weights.Span[i].Rows * gradients.Weights.Span[i].Columns; j++)
        {
            gradients.Weights.Span[i][j] /= (float)trainingCount;
        }
    }

    for (var i = 0; i < gradients.Bias.Length; i++)
    {
        for (var j = 0; j < gradients.Bias.Span[i].Rows * gradients.Bias.Span[i].Columns; j++)
        {
            gradients.Bias.Span[i][j] /= (float)trainingCount;
        }
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

void PrintSection(string name)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"===== {name} =====");
    Console.ForegroundColor = ConsoleColor.Gray;
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

var topology = new int[] { 2, 2, 1 };

var model = new NeuralNetwork(topology);
var gradients = new NeuralNetwork(topology);

var oldModel = new NeuralNetwork(topology);
var oldGradients = new NeuralNetwork(topology);

PrintSection("Training Data");
Console.WriteLine(trainingInput);
Console.WriteLine(trainingOutput);

var loss = Loss(trainingInput, trainingOutput, model);

PrintSection("Tensors");
Console.WriteLine(model);

Console.ForegroundColor = ConsoleColor.White;
Console.WriteLine($"Loss: {loss}");
Console.ForegroundColor = ConsoleColor.Gray;

// Training
const int trainingCount = 200;
Console.WriteLine($"Training for {trainingCount} iterations...");

for (var i = 0; i < trainingCount; i++)
{
    CalculateGradients(trainingInput, trainingOutput, oldModel, oldGradients);
    CalculateBackPropagation(trainingInput, trainingOutput, model, gradients);

    Learn(model, gradients);
    Learn(oldModel, oldGradients);
}

loss = Loss(trainingInput, trainingOutput, model);
var oldLoss = Loss(trainingInput, trainingOutput, oldModel);

PrintSection("After Training (Model)");
Console.WriteLine(model);

PrintSection("After Training (Old Model)");
Console.WriteLine(oldModel);

Console.ForegroundColor = ConsoleColor.White;
Console.WriteLine($"Loss: {loss}, Old Loss {oldLoss}");
Console.ForegroundColor = ConsoleColor.Gray;

PrintSection("Test");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;
    
        var inputs = new Tensor(1, 2, new float[] { x1, x2 });
        var y = model.Forward(inputs);

        Console.WriteLine($"{x1} ^ {x2} = {(int)MathF.Round(y[0])}");
    }
}
