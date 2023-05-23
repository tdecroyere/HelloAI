namespace TensorLib.UnitTests;

public class TensorTests
{
    [Fact]
    public void Multiply_DifferentDimensions_HasCorrectResults()
    {
        // Arrange
        var torch = new Torch();
        var tensor1 = torch.Tensor(new float[] { 1.5f, 2.5f }, 2, 1);
        var tensor2 = torch.Tensor(new float[] { 0.5f, 0.25f, 1.25f, 1.75f }, 2, 2);

        // Act
        var result = tensor1 * tensor2;

        // Assert
        Assert.Equal(tensor2.Size.X, result.Size.X);
        Assert.Equal(tensor1.Size.Y, result.Size.Y);
        Assert.Equal(tensor1[0, 0] * tensor2[0, 0] + tensor1[0, 1] * tensor2[1, 0], result[0, 0]);
        Assert.Equal(tensor1[0, 0] * tensor2[0, 1] + tensor1[0, 1] * tensor2[1, 1], result[0, 1]);
    }
}