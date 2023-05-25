namespace TensorLib.UnitTests;

public class TensorTests
{
    [Theory]
    [InlineData(2, 5)]
    [InlineData(5, 2)]
    [InlineData(0, 0)]
    public void Constructor_ShouldThrowArgumentOutOfRangeException_WhenWrongDimensionsArePassed(int rows, int columns)
    {
        // Arrange
        var data = new float[] { 3, 5, 5, 6 };

        // Act
        var action = () => { _ = new Tensor(rows, columns, data); };

        // Assert
        Assert.Throws<ArgumentOutOfRangeException>(action);
    }
    
    [Fact]
    public void Constructor_ShouldHaveModifiedData_WhenOtherTensorModifySameData()
    {
        // Arrange
        var tensor1 = new Tensor(1, 2, new float[] { 1.5f, 2.4f });     

        // Act
        var tensor2 = new Tensor(2, 1, tensor1);
        tensor1[0] = 5.0f;

        // Assert
        Assert.Equal(5.0f, tensor2[0]);
    }

    [Fact]
    public void Constructor_ShouldNotHaveModifiedData_WhenInitialDataIsModified()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.4f };

        // Act
        var tensor = new Tensor(2, 1, data);
        data[0] = 5.0f;

        // Assert
        Assert.NotEqual(5.0f, tensor[0]);
    }

    [Fact]
    public void Multiply_ShouldHaveCorrectResults_WhenDifferentDimensions()
    {
        // Arrange
        var tensor1 = new Tensor(1, 2, new float[] { 1.5f, 2.5f });
        var tensor2 = new Tensor(2, 2, new float[] { 0.5f, 0.25f, 1.25f, 1.75f });

        // Act
        var result = tensor1 * tensor2;

        // Assert
        Assert.Equal(result.Rows, tensor1.Rows);
        Assert.Equal(result.Columns, tensor2.Columns);
        Assert.Equal(tensor1[0, 0] * tensor2[0, 0] + tensor1[0, 1] * tensor2[1, 0], result[0, 0]);
        Assert.Equal(tensor1[0, 0] * tensor2[0, 1] + tensor1[0, 1] * tensor2[1, 1], result[0, 1]);
    }

    [Fact]
    public void Multiply_ShouldThrowArgumentOutOfRangeException_WhenIncompatibleDimensions()
    {
        // Arrange
        var tensor1 = new Tensor(2, 3);
        var tensor2 = new Tensor(5, 4);

        // Act
        var action = () => { _ = tensor1 * tensor2; };

        // Assert
        Assert.Throws<ArgumentOutOfRangeException>(action);
    }
}