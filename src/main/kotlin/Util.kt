import java.io.File
import java.util.Collections
import kotlin.math.*
import kotlin.random.Random

/**
 * Uses a Box-Muller transform for generating Gaussian random numbers.
 * Provides a way to generate random numbers that, when taken together, approximate a normal distribution with a mean
 * of 0 and a variance of 1.
 */
fun Random.nextGaussian(): Double {
    //Generates two random numbers
    val u = Random.nextDouble()
    val v = Random.nextDouble()
    //Return a sampling from the normal distribution.
    return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * kotlin.math.PI * v)
}

/**
 * Activation functions and their derivatives.
 */
fun relu(x: Double): Double = if (x > 0) x else 0.0
fun reluPrime(x: Double): Double = if (x > 0) 1.0 else 0.0

fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))
fun sigmoidPrime(x: Double): Double {
    val s = sigmoid(x)
    return s * (1 - s)
}

/**
 * Softmax activation function.
 */
fun softmax(x: DoubleArray): DoubleArray {
    val max = x.maxOrNull() ?: 0.0
    val exps = x.map { exp(it - max) }
    val sum = exps.sum()
    return exps.map { it / sum }.toDoubleArray()
}

/**
 * Cross–entropy loss function.
 * Assumes target is provided as a one–hot vector.
 */
fun crossEntropyLoss(predicted: DoubleArray, target: DoubleArray): Double {
    var loss = 0.0
    for (i in predicted.indices) {
        loss -= target[i] * ln(predicted[i] + 1e-15)
    }
    return loss
}

/**
 * Loads MNIST data from a CSV file.
 *
 * The CSV is expected to have 785 columns per row:
 * - The first column is the label (0–9).
 * - The remaining 784 columns are pixel values (0–255) for a 28×28 image.
 *
 * The pixel values are normalized to the range [0, 1].
 *
 * @param filePath The path to the CSV file.
 * @param skipHeader If true, skips the first line of the CSV file (assumed to be a header). Default is true.
 * @return A list of pairs where the first element is a 28×28 image (2D array of Doubles) and the second is a one-hot target vector.
 */
fun loadMNISTData(filePath: String, skipHeader: Boolean = true): List<Pair<Array<DoubleArray>, DoubleArray>> {
    val data = mutableListOf<Pair<Array<DoubleArray>, DoubleArray>>()
    val lines = File(filePath).readLines()
    val dataLines = if (skipHeader) lines.drop(1) else lines

    for (line in dataLines) {
        val tokens = line.split(",")
        // Ensure the line contains 785 values (1 label + 784 pixels)
        if (tokens.size != 785) continue

        // Parse the label and create a one-hot target vector.
        val label = tokens[0].toInt()
        val target = DoubleArray(10) { if (it == label) 1.0 else 0.0 }

        // Parse and normalize pixel values.
        val pixels = tokens.subList(1, tokens.size).map { it.toDouble() / 255.0 }
        // Reshape the flat list of 784 pixels into a 28x28 2D array.
        val image = Array(28) { row ->
            DoubleArray(28) { col ->
                pixels[row * 28 + col]
            }
        }
        data.add(Pair(image, target))
    }
    return data
}