/* =======================================================================================
   FLATTEN LAYER
   ======================================================================================= */

/**
 * A helper layer to flatten a 3D tensor into a 1D vector.
 */
class FlattenLayer {
    // Saved shape of the input tensor.
    var originalShape: Triple<Int, Int, Int>? = null

    /**
     * Forward pass.
     *
     * @param input 3D array (e.g. from the pooling layer).
     * @return Flattened 1D array.
     */
    fun forward(input: Array<Array<DoubleArray>>): DoubleArray {
        val numFilters = input.size
        val height = input[0].size
        val width = input[0][0].size
        originalShape = Triple(numFilters, height, width)
        val flatList = mutableListOf<Double>()
        for (f in 0 until numFilters) {
            for (i in 0 until height) {
                for (j in 0 until width) {
                    flatList.add(input[f][i][j])
                }
            }
        }
        return flatList.toDoubleArray()
    }

    /**
     * Backward pass.
     *
     * @param gradOutput Gradient as a flat 1D array.
     * @return Reshaped gradient matching the original 3D input shape.
     */
    fun backward(gradOutput: DoubleArray): Array<Array<DoubleArray>> {
        val (numFilters, height, width) = originalShape!!
        val output = Array(numFilters) { Array(height) { DoubleArray(width) { 0.0 } } }
        var index = 0
        for (f in 0 until numFilters) {
            for (i in 0 until height) {
                for (j in 0 until width) {
                    output[f][i][j] = gradOutput[index++]
                }
            }
        }
        return output
    }
}