using System.Numerics;
using System.Text;

var trainingData = new float[,]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 1.0f }
};


float LossFunction(float w1, float w2, float b)
{
    var result = 0.0f;

    for (var i = 0; i < trainingData.GetLength(0); i++)
    {
        var x1 = trainingData[i, 0];
        var x2 = trainingData[i, 1];

        var y = SigmoidF(x1 * w1 + x2 * w2 + b);
        var testResult = trainingData[i, 2];

        var d = y - testResult;
        result += d * d;
    }

    result /= trainingData.GetLength(0);
    return result;
}

float LossFunctionTensor(Torch torch, Tensor weights, Tensor bias)
{
    var result = 0.0f;

    for (var j = 0; j < trainingData.GetLength(0); j++)
    {
        var x1 = trainingData[j, 0];
        var x2 = trainingData[j, 1];
        var testResult = trainingData[j, 2];

        var input = torch.Tensor(new float[] { x1, x2 }, 1, 2);
        var y = Sigmoid(input * weights + bias);

        var d = y[0] - testResult;
        result += d * d;
    }

    result /= trainingData.GetLength(0);
    return result;
}

float SigmoidF(float x)
{
    return 1.0f / (1.0f + MathF.Exp(-x));
}

Tensor Sigmoid(Tensor x)
{
    // TODO: Create an exp method in torch
    var result = new Tensor(x.ElementCount, 1);

    for (var i = 0; i < x.ElementCount; i++)
    {
        result[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
    }
    return result;
}

var random = new Random(28);

var epsilon = 1e-3f;
var learningRate = 0.01f;

var w1 = random.NextSingle() * 10.0f;
var w2 = random.NextSingle() * 10.0f;
var b = random.NextSingle() * 5.0f;

var testLoss = LossFunction(w1, w2, b);
Console.WriteLine($"w1: {w1}, w2: {w2}, b: {b}, Loss: {testLoss}");

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = LossFunction(w1, w2, b);

    var dw1 = (LossFunction(w1 + epsilon, w2, b) - loss) / epsilon;
    var dw2 = (LossFunction(w1, w2 + epsilon, b) - loss) / epsilon;
    var db = (LossFunction(w1, w2, b + epsilon) - loss) / epsilon;

    w1 -= dw1 * learningRate;
    w2 -= dw2 * learningRate;
    b -= db * learningRate;

    //Console.WriteLine($"w1: {w1}, w2: {w2}, b: {b}, Loss: {loss}");

}

var finalLoss = LossFunction(w1, w2, b);
Console.WriteLine($"w1: {w1}, w2: {w2}, b: {b}, Loss: {finalLoss}");

// Test
var torch = new Torch();

//var inputLayer = new Tensor(2, 2);
var outputLayerWeights = torch.Random(2, 1) * 10.0f;
var outputLayerBias = torch.Random(1, 1) * 5.0f;

Console.WriteLine("===== Tensors =====");

var lossTensor = LossFunctionTensor(torch, outputLayerWeights, outputLayerBias);
Console.WriteLine($"w: {outputLayerWeights} b: {outputLayerBias}, Loss: {lossTensor}");

// Training
for (var i = 0; i < 100000; i++)
{
    var loss = LossFunctionTensor(torch, outputLayerWeights, outputLayerBias);
    // HACK: Really slow!
    var copyWeight = outputLayerWeights.Copy();

    // HACK: Really slow!
    for (var j = 0; j < outputLayerWeights.ElementCount; j++)
    {
        var copy = outputLayerWeights.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(torch, copy, outputLayerBias) - loss) / epsilon;
        outputLayerWeights[j] -= diff * learningRate;
    }
    
    for (var j = 0; j < outputLayerBias.ElementCount; j++)
    {
        var copy = outputLayerBias.Copy();
        copy[j] += epsilon;
        var diff = (LossFunctionTensor(torch, copyWeight, copy) - loss) / epsilon;
        outputLayerBias[j] -= diff * learningRate;
    }

    //Console.WriteLine($"w: {outputLayerWeights} b: {outputLayerBias}, Loss: {loss}");
}

var finalLossTensor = LossFunctionTensor(torch, outputLayerWeights, outputLayerBias);
Console.WriteLine($"w: {outputLayerWeights} b: {outputLayerBias}, Loss: {finalLossTensor}");

Console.WriteLine("===== TEST =====");

for (var i = 0; i < 2; i++)
{
    for (var j = 0; j < 2; j ++)
    {
        var x1 = (float)i;
        var x2 = (float)j;

        var input = torch.Tensor(new float[] { x1, x2 }, 1, 2);
        var y = Sigmoid(input * outputLayerWeights + outputLayerBias);

        Console.WriteLine($"Test calcul: {x1} | {x2} = {y}");
    }
}

public readonly record struct Tensor
{
    private readonly float[] _data;

    internal Tensor(int width, int height)
    {
        Size = new Vector2(width, height);
        _data = new float[width * height];
    }
    
    internal Tensor(Span<float> data, int width, int height)
    {
        Size = new Vector2(width, height);
        _data = new float[data.Length];
        data.CopyTo(_data);
    }

    public Vector2 Size { get; }
    public int ElementCount => _data.Length; 

    public float this[int index]
    {
        get
        {
            return _data[index];
        }
        set
        {
            _data[index] = value;
        }
    }

    public Tensor Copy()
    {
        return new Tensor(_data, (int)Size.X, (int)Size.Y);
    }

    public override string ToString()
    {
        var output = new StringBuilder();

        output.Append("Tensor(");

        if (Size.Y > 1)
        {
            output.Append('[');
        }

        var padding = output.Length;

        for (var i = 0; i < Size.Y; i++)
        {
            output.Append('[');

            for (var j = 0; j < Size.X; j++)
            {
                if (j > 0)
                {
                    output.Append(',');
                }

                if (j > 0)
                {
                    if (j % 8 == 0)
                    {
                        output.AppendLine("");

                        for (var k = 0; k < padding; k++)
                        {
                            output.Append(" ");
                        }
                    }
                    else
                    {
                        output.Append('\t');
                    }
                }

                output.Append(_data[i * (int)Size.X + j].ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            if (i < Size.Y - 1)
            {
                output.AppendLine("],");
                output.Append("        ");
            }
            else
            {
                output.Append(']');
            }
        }

        if (Size.Y > 1)
        {
            output.Append(']');
        }

        output.Append(')');

        return output.ToString();
    }

    public static Tensor operator*(Tensor tensor, float scalar)
    {
        return Multiply(tensor, scalar);
    }

    public static Tensor Multiply(Tensor tensor, float scalar)
    {
        for (var i = 0; i < tensor._data.Length; i++)
        {
            tensor._data[i] *= scalar;
        }
        
        return tensor;
    }
    
    public static Tensor operator*(Tensor tensor1, Tensor tensor2)
    {
        return Multiply(tensor1, tensor2);
    }

    public static Tensor Multiply(Tensor tensor1, Tensor tensor2)
    {
        // TODO: Check compatibility
        // TODO: Avoid this allocation
        var size = new Vector2(tensor1.Size.X, tensor2.Size.Y);
        var resultData = new float[(int)size.X * (int)size.Y];

        for (var i = 0; i < (int)size.X; i++)
        {
            for (var j = 0; j < (int)size.Y; j++)
            {
                var result = 0.0f;

                // TODO: Iterate with common dim
                for (var k = 0; k < (int)tensor1.Size.Y; k++)
                {
                    var element1 = tensor1._data[j * (int)size.Y + k];
                    var element2 = tensor2._data[j * (int)size.X + k];
                    result += element1 * element2;
                }

                resultData[i] = result;
            }
        }
        
        return new Tensor(resultData, (int)size.X, (int)size.Y);
    }

    public static Tensor operator+(Tensor tensor1, Tensor tensor2)
    {
        return Add(tensor1, tensor2);
    }

    public static Tensor Add(Tensor tensor1, Tensor tensor2)
    {
        // TODO: Check compatibility
        
        var resultData = new float[tensor1._data.Length];

        // HACK: For now we only use the first element to make the bias work :)
        for (var i = 0; i < tensor1._data.Length; i++)
        {
            resultData[i] = tensor1._data[i] + tensor2._data[0];
        }
        
        // TODO: Change that
        return new Tensor(resultData, resultData.Length, 1);
    }
}

public class Torch
{
    private Random _random = new(28);

    public Tensor Random(int width, int height)
    {
        var result = new Tensor(width, height);

        for (var i = 0; i < result.ElementCount; i++)
        {
            result[i] = _random.NextSingle();
        }

        return result;
    }

    public Tensor Tensor(float[] data)
    {
        return new Tensor(data, data.Length, 1);
    }
    
    public Tensor Tensor(float[] data, int width, int height)
    {
        return new Tensor(data, width, height);
    }
}