import java.io.File

/* =======================================================================================
   SIMPLE CNN NETWORK
   ======================================================================================= */

/**
 * A simple CNN composed of:
 * - A convolution layer (with ReLU activation)
 * - A max–pooling layer
 * - A flatten layer
 * - A dense layer followed by softmax output
 *
 * The architecture is designed for MNIST images (28x28 grayscale).
 */
class SimpleCNN {
    // Convolution: with 8 filters of size 3×3, stride 1.
    val conv = ConvolutionLayer(numFilters = 8, filterSize = 3, stride = 1)
    // Pooling: 2×2 max pooling.
    val pool = MaxPoolingLayer(poolSize = 2)
    // Flatten: will convert the pooled output to a 1D vector.
    val flatten = FlattenLayer()
    // Dense: input size = 8 filters × 13×13 (since 28→26 in conv then /2 in pooling) = 1352, output = 10 classes.
    val dense = DenseLayer(inputSize = 1352, outputSize = 10)

    /**
     * Forward propagation through the network.
     *
     * @param input A 28×28 image represented as a 2D array.
     * @return A pair where the first element is the softmax–activated output vector
     *         and the second element is the dense output (pre–softmax) used for backprop.
     */
    fun forward(input: Array<DoubleArray>): Pair<DoubleArray, DoubleArray> {
        // Convolution layer forward.
        val convOut = conv.forward(input)              // shape: [8][26][26]
        // Max pooling forward.
        val poolOut = pool.forward(convOut)              // shape: [8][13][13]
        // Flatten.
        val flatOut = flatten.forward(poolOut)           // 1D vector (length 1352)
        // Dense layer forward.
        val denseOut = dense.forward(flatOut)            // 1D vector (length 10)
        // Softmax for classification.
        val output = softmax(denseOut)
        return Pair(output, denseOut)
    }

    /**
     * Backward propagation through the network.
     *
     * @param gradLoss Gradient of the loss with respect to the softmax output.
     *                 (For cross–entropy with softmax, this is [predicted - target]).
     * @param learningRate Learning rate for parameter updates.
     */
    fun backward(gradLoss: DoubleArray, learningRate: Double) {
        // Backprop through dense layer.
        val gradDense = dense.backward(gradLoss)  // Gradient w.r.t. flattened input.
        dense.updateParams(learningRate)
        // Backprop through flatten layer.
        val gradPool = flatten.backward(gradDense)  // Reshape to [8][13][13].
        // Backprop through pooling layer.
        val gradConv = pool.backward(gradPool)        // Gradient w.r.t. conv output.
        // Backprop through convolution layer.
        conv.backward(gradConv)
        conv.updateParams(learningRate)
    }

    /**
     * Train the network on a single example.
     *
     * @param input 28×28 input image.
     * @param target One–hot encoded target vector (length 10).
     * @param learningRate Learning rate.
     * @return The computed loss for this example.
     */
    fun trainExample(input: Array<DoubleArray>, target: DoubleArray, learningRate: Double): Double {
        val (output, denseOut) = forward(input)
        val loss = crossEntropyLoss(output, target)
        // For cross–entropy loss with softmax, the gradient is (output - target).
        val gradLoss = DoubleArray(output.size) { i -> output[i] - target[i] }
        backward(gradLoss, learningRate)
        return loss
    }

    /**
     * Predict the class label for a given input image.
     *
     * @param input 28×28 input image.
     * @return Predicted label (0–9).
     */
    fun predict(input: Array<DoubleArray>): Int {
        val (output, _) = forward(input)
        return output.indices.maxByOrNull { output[it] } ?: -1
    }

    /**
     * Save model parameters to a file.
     *
     * @param filePath Path to the file where the model will be saved.
     */
    fun saveModel(filePath: String) {
        File(filePath).printWriter().use { out ->
            out.println("ConvFilters")
            for (f in conv.filters) {
                for (row in f) {
                    out.println(row.joinToString(","))
                }
                out.println("EndFilter")
            }
            out.println("ConvBiases")
            out.println(conv.biases.joinToString(","))
            out.println("DenseWeights")
            for (row in dense.weights) {
                out.println(row.joinToString(","))
            }
            out.println("DenseBiases")
            out.println(dense.biases.joinToString(","))
        }
    }

    /**
     * Load model parameters from a file.
     *
     * @param filePath Path to the file from which the model parameters are loaded.
     */
    fun loadModel(filePath: String) {
        val lines = File(filePath).readLines()
        var index = 0
        if (lines[index++] != "ConvFilters") return
        // Load convolution filters (assumes numFilters filters each of filterSize lines, ending with "EndFilter").
        for (f in 0 until conv.numFilters) {
            for (i in 0 until conv.filterSize) {
                val rowValues = lines[index++].split(",").map { it.toDouble() }
                conv.filters[f][i] = rowValues.toDoubleArray()
            }
            index++ // Skip "EndFilter"
        }
        if (lines[index++] != "ConvBiases") return
        conv.biases = lines[index++].split(",").map { it.toDouble() }.toDoubleArray()
        if (lines[index++] != "DenseWeights") return
        for (i in 0 until dense.outputSize) {
            val rowValues = lines[index++].split(",").map { it.toDouble() }
            dense.weights[i] = rowValues.toDoubleArray()
        }
        if (lines[index++] != "DenseBiases") return
        dense.biases = lines[index++].split(",").map { it.toDouble() }.toDoubleArray()
    }
}