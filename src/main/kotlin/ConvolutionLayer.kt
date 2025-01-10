import kotlin.random.Random

/* =======================================================================================
   CONVOLUTION LAYER
   ======================================================================================= */

/**
 * A simple convolution layer for a single–channel (grayscale) image.
 *
 * @param numFilters Number of convolution filters.
 * @param filterSize Size (width and height) of each square filter.
 * @param stride Stride of the convolution.
 */
class ConvolutionLayer(val numFilters: Int, val filterSize: Int, val stride: Int = 1) {

    // Each filter is a 2D kernel. Filters: [filterIndex][row][col]
    var filters: Array<Array<DoubleArray>> = Array(numFilters) {
        Array(filterSize) { DoubleArray(filterSize) { Random.nextGaussian() } }
    }
    // One bias per filter.
    var biases: DoubleArray = DoubleArray(numFilters) { Random.nextGaussian() }

    // Saved input and pre–activation values (z) from the forward pass.
    lateinit var input: Array<DoubleArray>
    lateinit var lastZ: Array<Array<DoubleArray>>

    // Gradients to be computed during backprop.
    lateinit var gradFilters: Array<Array<DoubleArray>>
    lateinit var gradBiases: DoubleArray

    /**
     * Forward pass.
     *
     * @param input A 2D array representing the grayscale input image.
     * @return 3D array of feature maps with shape [numFilters][outputHeight][outputWidth].
     */
    fun forward(input: Array<DoubleArray>): Array<Array<DoubleArray>> {
        this.input = input
        val inputHeight = input.size
        val inputWidth = input[0].size
        val outputHeight = (inputHeight - filterSize) / stride + 1
        val outputWidth = (inputWidth - filterSize) / stride + 1

        // z stores the pre–activation (convolution sum + bias) values.
        val z = Array(numFilters) { Array(outputHeight) { DoubleArray(outputWidth) } }
        // Output applies the ReLU activation.
        val output = Array(numFilters) { Array(outputHeight) { DoubleArray(outputWidth) } }

        for (f in 0 until numFilters) {
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    var sum = 0.0
                    // Convolve filter over the region.
                    for (ki in 0 until filterSize) {
                        for (kj in 0 until filterSize) {
                            sum += input[i * stride + ki][j * stride + kj] * filters[f][ki][kj]
                        }
                    }
                    sum += biases[f]
                    z[f][i][j] = sum
                    output[f][i][j] = relu(sum)
                }
            }
        }
        lastZ = z
        return output
    }

    /**
     * Backward pass.
     *
     * @param gradOutput Gradient of loss with respect to the layer's output.
     *                   Has shape [numFilters][outputHeight][outputWidth].
     * @return Gradient with respect to the input image.
     */
    fun backward(gradOutput: Array<Array<DoubleArray>>): Array<DoubleArray> {
        val inputHeight = input.size
        val inputWidth = input[0].size
        val outputHeight = gradOutput[0].size
        val outputWidth = gradOutput[0][0].size

        val gradInput = Array(inputHeight) { DoubleArray(inputWidth) { 0.0 } }
        // Initialize gradients for filters and biases.
        gradFilters = Array(numFilters) { Array(filterSize) { DoubleArray(filterSize) { 0.0 } } }
        gradBiases = DoubleArray(numFilters) { 0.0 }

        for (f in 0 until numFilters) {
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    // Compute derivative of ReLU: only backpropagate where z > 0.
                    val dZ = gradOutput[f][i][j] * reluPrime(lastZ[f][i][j])
                    gradBiases[f] += dZ
                    for (ki in 0 until filterSize) {
                        for (kj in 0 until filterSize) {
                            gradFilters[f][ki][kj] += input[i * stride + ki][j * stride + kj] * dZ
                            gradInput[i * stride + ki][j * stride + kj] += filters[f][ki][kj] * dZ
                        }
                    }
                }
            }
        }
        return gradInput
    }

    /**
     * Update the filter weights and biases with the computed gradients.
     *
     * @param learningRate Learning rate for the update.
     */
    fun updateParams(learningRate: Double) {
        for (f in 0 until numFilters) {
            for (i in 0 until filterSize) {
                for (j in 0 until filterSize) {
                    filters[f][i][j] -= learningRate * gradFilters[f][i][j]
                }
            }
            biases[f] -= learningRate * gradBiases[f]
        }
    }
}