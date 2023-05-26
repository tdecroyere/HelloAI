namespace TensorLib;

// TODO: Memory management?

public readonly record struct Tensor
{
    private readonly Memory<float> _data;
    private readonly int _stride;

    public Tensor(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;

        _data = new float[rows * columns];
        _stride = 0;
    }

    public Tensor(int rows, int columns, Func<int, int, float> generator) : this(rows, columns)
    {
        ArgumentNullException.ThrowIfNull(generator);

        for (var i = 0; i < ElementCount; i++)
        {
            _data.Span[i] = generator(i, i);
        }
    }

    public Tensor(int rows, int columns, Memory<float> data)
    {
        if (rows * columns != data.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(data), "Data element count is not rows * columns.");
        }

        Rows = rows;
        Columns = columns;

        _data = new float[data.Length];
        data.CopyTo(_data);
        _stride = 0;
    }

    private Tensor(int rows, int columns, int offset, int stride, Tensor source)
    {
        Rows = rows;
        Columns = columns;
        _data = source._data[offset..];
        _stride = stride;
    }

    public int Rows { get; }
    public int Columns { get; }
    public int ElementCount => _data.Length; 

    public float this[int index]
    {
        get
        {
            var rowIndex = index / Columns;
            var columnIndex = index % Columns;
        
            return this[rowIndex, columnIndex];
        }
        set
        {
            var rowIndex = index / Columns;
            var columnIndex = index % Columns;
            
            this[rowIndex, columnIndex] = value;
        }
    }

    public float this[int rowIndex, int columnIndex]
    {
        get
        {
            return _data.Span[rowIndex * (Columns + _stride) + columnIndex];
        }
        set
        {
            _data.Span[rowIndex * (Columns + _stride) + columnIndex] = value;
        }
    }

    public Tensor View(int rows, int columns)
    {
        return new Tensor(rows, columns, 0, Columns - columns, this);
    }

    public Tensor View(int rows, int columns, int columnOffset)
    {
        return new Tensor(rows, columns, columnOffset, 1 + Columns - columnOffset, this);
    }

    public Tensor RowView(int rowIndex)
    {
        return new Tensor(1, Columns, rowIndex * (Columns + _stride), _stride, this);
    }

    public Tensor Copy()
    {
        // TODO: If we copy a "view" copy only the necessary data

        var newData = new float[_data.Length];
        _data.CopyTo(newData);

        return new Tensor(Rows, Columns, newData);
    }

    public override string ToString()
    {
        var output = new StringBuilder();

        if (Rows > 1)
        {
            output.AppendLine();
        }

        output.Append("Tensor(");

        if (Rows > 1)
        {
            output.Append('[');
        }

        var padding = output.Length;

        for (var i = 0; i < Rows; i++)
        {
            output.Append('[');

            for (var j = 0; j < Columns; j++)
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
                            output.Append(' ');
                        }
                    }
                    else
                    {
                        output.Append('\t');
                    }
                }

                output.Append(this[i, j].ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            if (i < Rows - 1)
            {
                output.AppendLine("],");
                output.Append("        ");
            }
            else
            {
                output.Append(']');
            }
        }

        if (Rows > 1)
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
        if (tensor1.Columns != tensor2.Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(tensor1), "Tensor1.Columns should be equal to tensor2.Rows");
        }

        var resultData = new Tensor(tensor1.Rows, tensor2.Columns);

        for (var i = 0; i < resultData.Rows; i++)
        {
            for (var j = 0; j < resultData.Columns; j++)
            {
                var result = 0.0f;

                // TODO: Iterate with common dim
                for (var k = 0; k < tensor1.Columns; k++)
                {
                    result += tensor1[i, k] * tensor2[k, j];
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
        var resultData = new Tensor(tensor1.Rows, tensor1.Columns); 

        // HACK: For now we only use the first element to make the bias work :)
        for (var i = 0; i < tensor1._data.Length; i++)
        {
            resultData[i] = tensor1._data.Span[i] + tensor2._data.Span[i];
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