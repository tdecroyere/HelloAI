namespace TorchLib;

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

    public float this[int row, int column]
    {
        get
        {
            return _data[row * (int)Size.X + column];
        }
        set
        {
            _data[row * (int)Size.X + column] = value;
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
        var result = tensor.Copy();

        for (var i = 0; i < tensor.ElementCount; i++)
        {
            result[i] *= scalar;
        }
        
        return result;
    }
    
    public static Tensor operator*(Tensor tensor1, Tensor tensor2)
    {
        return Multiply(tensor1, tensor2);
    }

    public static Tensor Multiply(Tensor tensor1, Tensor tensor2)
    {
        // TODO: Check compatibility
        // TODO: Avoid this allocation
        var size = new Vector2(tensor2.Size.X, tensor1.Size.Y);
        var resultData = new Tensor((int)size.X, (int)size.Y);

        for (var i = 0; i < (int)size.Y; i++)
        {
            for (var j = 0; j < (int)size.X; j++)
            {
                var result = 0.0f;

                // TODO: Iterate with common dim
                for (var k = 0; k < (int)tensor1.Size.X; k++)
                {
                    var element1 = tensor1[i, k];
                    var element2 = tensor2[k, j];
                    result += element1 * element2;
                }

                resultData[i, j] = result;
            }
        }
        
        return resultData;
    }

    public static Tensor operator+(Tensor tensor1, Tensor tensor2)
    {
        return Add(tensor1, tensor2);
    }

    public static Tensor Add(Tensor tensor1, Tensor tensor2)
    {
        // TODO: Check compatibility
        var resultData = new Tensor((int)tensor1.Size.X, (int)tensor1.Size.Y); 

        // HACK: For now we only use the first element to make the bias work :)
        for (var i = 0; i < tensor1._data.Length; i++)
        {
            resultData[i] = tensor1._data[i] + tensor2._data[i];
        }
        
        // TODO: Change that
        return resultData;
    }
    
    public static Tensor operator-(Tensor tensor, float scalar)
    {
        return Subtract(tensor, scalar);
    }

    public static Tensor Subtract(Tensor tensor, float scalar)
    {
        var result = tensor.Copy();

        for (var i = 0; i < tensor.ElementCount; i++)
        {
            result[i] -= scalar;
        }
        
        return result;
    }
    
}