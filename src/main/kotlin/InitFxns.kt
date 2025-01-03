import kotlin.random.Random

// Utility
fun generateBiases(neuronsPerLayer: List<Int>): List<List<Double>> {
    //Then, we transform the list, element by element, morphing each element into a respective list
    //with a randomly generated gaussian.
    //The larger 'sizes' is, the distribution of the samples from nextGaussian() will start to resemble
    //the bell-shaped curve characteristic of a normal distribution.
    return neuronsPerLayer.drop(1).map { y ->
        List(y) { Random.nextGaussian() }
    }
}

// Generate weights for all layers, connecting neurons between layers.
fun generateWeights(neuronsPerLayer: List<Int>): List<List<List<Double>>> {
    // Create a 3D list where each inner list represents the weights for connections to neurons in a layer.
    /*
    The zip produces a list of tuples the represents connection between layers. (first layer amnt, second layer amnt)
    The map takes each element from the tuple list and replaces the considered element with a double list, where
    the outer list is the second element of the considered tuple, and the inner list is a list of gaussians.

    So say we have input as (a, b, c)
    zipped = input.dropLast(1).zip(input.drop(1)) = ((a, b) (b, c))

    zipped.map(){(x, y) -> List(y) {List (x) {random.nextGaussian()}}}

    would look like:
    [{(), ()}, {(), ()}]

     */
    return neuronsPerLayer.dropLast(1).zip(neuronsPerLayer.drop(1)).map { (x, y) ->
        //e.g. if (x, y) represents (last_hidden_layer_neuronAmnt, output_layer_neuronAmnt)
        //Then the last element in this list will be a List of the amnt of output neurons, where each list corresponding to each of the
        //output neurons (dictated by the amount of neurons in previous layer, because the # of weights is dictated by such)

        //Basically: List<weightAmnt from previous layer<Neuron amount in currently considered layer, which will be filled with generated gaussians>>
        List(y) { List(x) { Random.nextGaussian() } }
    }
}

/*
Uses a Box-Muller transform for generating Gaussian random numbers.
Provides a way to generate random numbers that, when taken together, approximate a normal distribution with a mean
of 0 and a variance of 1.
 */
fun Random.nextGaussian(): Double {
    //Generates two random numbers
    val u = Random.nextDouble()
    val v = Random.nextDouble()
    //Return a sampling from the normal distribution.
    return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * kotlin.math.PI * v)
}