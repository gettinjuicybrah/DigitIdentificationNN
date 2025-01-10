/* =======================================================================================
   MAX–POOLING LAYER
   ======================================================================================= */

/**
 * A simple max pooling layer.
 *
 * @param poolSize Size of the (square) pooling window.
 */
class MaxPoolingLayer(val poolSize: Int) {

    // Saved input and indices of maximum values for backprop.
    lateinit var input: Array<Array<DoubleArray>>
    // For each output element we save the (row, col) index in the input that had the maximum value.
    lateinit var maxIndices: Array<Array<Array<Pair<Int, Int>>>>

    /**
     * Forward pass.
     *
     * @param input 3D array from the convolution layer with shape [numFilters][height][width].
     * @return Downsampled feature maps.
     */
    fun forward(input: Array<Array<DoubleArray>>): Array<Array<DoubleArray>> {
        this.input = input
        val numFilters = input.size
        val inputHeight = input[0].size
        val inputWidth = input[0][0].size
        val outputHeight = inputHeight / poolSize
        val outputWidth = inputWidth / poolSize
        val output = Array(numFilters) { Array(outputHeight) { DoubleArray(outputWidth) } }
        maxIndices = Array(numFilters) {
            Array(outputHeight) { Array(outputWidth) { Pair(0, 0) } }
        }

        for (f in 0 until numFilters) {
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    var maxVal = Double.NEGATIVE_INFINITY
                    var maxIdx = Pair(0, 0)
                    for (m in 0 until poolSize) {
                        for (n in 0 until poolSize) {
                            val row = i * poolSize + m
                            val col = j * poolSize + n
                            if (input[f][row][col] > maxVal) {
                                maxVal = input[f][row][col]
                                maxIdx = Pair(row, col)
                            }
                        }
                    }
                    output[f][i][j] = maxVal
                    maxIndices[f][i][j] = maxIdx
                }
            }
        }
        return output
    }

    /**
     * Backward pass.
     *
     * @param gradOutput Gradient with respect to the output of the pooling layer.
     * @return Gradient with respect to the input of the pooling layer.
     */
    fun backward(gradOutput: Array<Array<DoubleArray>>): Array<Array<DoubleArray>> {
        val numFilters = input.size
        val inputHeight = input[0].size
        val inputWidth = input[0][0].size
        val gradInput = Array(numFilters) { Array(inputHeight) { DoubleArray(inputWidth) { 0.0 } } }
        val outputHeight = gradOutput[0].size
        val outputWidth = gradOutput[0][0].size

        for (f in 0 until numFilters) {
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    val (maxRow, maxCol) = maxIndices[f][i][j]
                    gradInput[f][maxRow][maxCol] = gradOutput[f][i][j]
                }
            }
        }
        return gradInput
    }
}