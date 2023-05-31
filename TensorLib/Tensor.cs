namespace TensorLib;

// TODO: Memory management?
// TODO: Review each memory copy and look at perf!

public readonly record struct Tensor
{
    private readonly Memory<float> _data;

    private readonly int _rowStride;
    private readonly int _offset;

    public Tensor(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;

        _data = new float[rows * columns];
        _rowStride = columns;
        _offset = 0;
    }

    public Tensor(int rows, int columns, Func<int, int, float> generator) : this(rows, columns)
    {
        ArgumentNullException.ThrowIfNull(generator);

        for (var i = 0; i < rows * columns; i++)
        {
            _data.Span[i] = generator(i, i);
        }
    }

    public Tensor(int rows, int columns, Memory<float> data) : this(rows, columns)
    {
        if (rows * columns != data.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(data), "Data element count is not rows * columns.");
        }
       
        data.CopyTo(_data);
    }

    private Tensor(int rows, int columns, int offset, int rowStride, Tensor source)
    {
        Rows = rows;
        Columns = columns;
        _offset = offset;
        _rowStride = rowStride;
        _data = source._data;
    }

    public int Rows { get; }
    public int Columns { get; }

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
            return _data.Span[_offset + rowIndex * _rowStride + columnIndex];
        }
        set
        {
            _data.Span[_offset + rowIndex * _rowStride + columnIndex] = value;
        }
    }

    public void Zero()
    {
        for (var i = 0; i < Rows * Columns; i++)
        {
            this[i] = 0.0f;
        }
    }

    public Tensor View(int rows, int columns)
    {
        return new Tensor(rows, columns, 0, Columns, this);
    }

    public Tensor View(int rows, int columns, int columnOffset)
    {
        return new Tensor(rows, columns, columnOffset, Columns, this);
    }

    public Tensor ViewRow(int rowIndex)
    {
        return new Tensor(1, Columns, _offset + rowIndex * _rowStride, _rowStride, this);
    }

    public Tensor Copy()
    {
        var dataCopy = new float[Rows * Columns];

        for (var i = 0; i < Rows * Columns; i++)
        {
            dataCopy[i] = this[i];
        }

        return new Tensor(Rows, Columns, dataCopy);
    }

    public override string ToString()
    {
        return ToString("Tensor");
    }

    public string ToString(string name)
    {
        var output = new StringBuilder();

        if (Rows > 1)
        {
            output.AppendLine();
        }

        output.Append($"{name}(");

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

                    if (j % 8 == 0)
                    {
                        output.AppendLine();

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

        for (var i = 0; i < tensor.Rows * tensor.Columns; i++)
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

        for (var i = 0; i < tensor1.Rows * tensor1.Columns; i++)
        {
            resultData[i] = tensor1[i] + tensor2[i];
        }
        
        return resultData;
    }
    
    public static Tensor operator-(Tensor tensor, float scalar)
    {
        return Subtract(tensor, scalar);
    }

    public static Tensor Subtract(Tensor tensor, float scalar)
    {
        var result = tensor.Copy();

        for (var i = 0; i < tensor.Rows * tensor.Columns; i++)
        {
            result[i] -= scalar;
        }
        
        return result;
    }
}
