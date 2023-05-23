namespace TensorLib;

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

    public Tensor Zeros(int width, int height)
    {
        return new Tensor(width, height);
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