import kotlin.random.Random

/* =======================================================================================
   DENSE (FULLY CONNECTED) LAYER
   ======================================================================================= */

/**
 * A simple dense (fully connected) layer.
 *
 * @param inputSize Number of input neurons.
 * @param outputSize Number of output neurons.
 */
class DenseLayer(val inputSize: Int, val outputSize: Int) {
    var weights: Array<DoubleArray> =
        Array(outputSize) { DoubleArray(inputSize) { Random.nextGaussian() } }
    var biases: DoubleArray = DoubleArray(outputSize) { Random.nextGaussian() }
    lateinit var input: DoubleArray

    // Gradients computed in the backward pass.
    lateinit var gradWeights: Array<DoubleArray>
    lateinit var gradBiases: DoubleArray

    /**
     * Forward pass.
     *
     * @param input A 1D input vector.
     * @return The computed output vector.
     */
    fun forward(input: DoubleArray): DoubleArray {
        this.input = input
        val output = DoubleArray(outputSize) { i ->
            var sum = biases[i]
            for (j in 0 until inputSize) {
                sum += weights[i][j] * input[j]
            }
            sum
        }
        return output
    }

    /**
     * Backward pass.
     *
     * @param gradOutput Gradient of the loss with respect to this layer's output.
     * @return Gradient with respect to the input of the layer.
     */
    fun backward(gradOutput: DoubleArray): DoubleArray {
        val gradInput = DoubleArray(inputSize) { 0.0 }
        gradWeights = Array(outputSize) { DoubleArray(inputSize) { 0.0 } }
        gradBiases = DoubleArray(outputSize) { 0.0 }

        for (i in 0 until outputSize) {
            gradBiases[i] = gradOutput[i]
            for (j in 0 until inputSize) {
                gradWeights[i][j] = gradOutput[i] * input[j]
                gradInput[j] += weights[i][j] * gradOutput[i]
            }
        }
        return gradInput
    }

    /**
     * Update the parameters using gradient descent.
     *
     * @param learningRate Learning rate for the update.
     */
    fun updateParams(learningRate: Double) {
        for (i in 0 until outputSize) {
            for (j in 0 until inputSize) {
                weights[i][j] -= learningRate * gradWeights[i][j]
            }
            biases[i] -= learningRate * gradBiases[i]
        }
    }
}