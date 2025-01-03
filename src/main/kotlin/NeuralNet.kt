import java.util.Collections
import kotlin.random.Random
import kotlin.math.*
import kotlin.properties.Delegates

/*
neuronsPerLayer indicates the # neurons per layer, including the input and output layers.
 */
class NeuralNet(neuronsPerLayer: List<Int>) {

    val layerAmnt = neuronsPerLayer.size
    //more efficient? Why or why not
    //Drop the first list element because input neurons will not have biases.
    val neuronsPerLayer = neuronsPerLayer
    val biasesList = generateBiases(neuronsPerLayer)
    val weightsList = generateWeights(neuronsPerLayer)

    /*
    PURPOSE: Get output from the network, given an input.
     */
    fun forwardPass(inputLayer: List<Double>): List<Double>{
        var input = inputLayer
        var activeList = mutableListOf<Any>()
        var currentLayer = 0
        while (currentLayer < layerAmnt) {
            for ((weights, bias) in weightsList[currentLayer].zip(biasesList[currentLayer])){
                activeList.add(dotProduct(input, weights) + bias)
            }
            input = input.map{element -> sigmoid(element)}
            currentLayer++
        }
        return input
    }

/*
Training data is a list of pairs that contains input data and the expected output data.

Learning rate: The influence a derivative will have on a cost function. 'How big of a step' it will take. The influence will ALWAYS be a decrease in the cost function output -
it's just a question of HOW MUCH a particular 'step' will influence the cost function's change (And in the context of talking about the cost function, at least here,
the CHANGE is always a DECREASE

We don't want the learning rate to be too big, and we don't want it to be too small because it would take too long.

Gradient Descent algorithm: Repeatedly compute the gradiant of the cost function and them move in the opposite direction to converge to converge to a minimum (hopefully global).



e

 */
    fun SGD(trainingData: List<Pair<List<Double>, List<Double>>>, epochs: Int, miniBatchSize: Int, learningRate: Int){
        val length = trainingData.size
        val range = 0..epochs

        for (j in range){
            Collections.shuffle(trainingData)
            val miniBatches = listOf()
        }

        for (j in range step miniBatchSize){
            Collections.shuffle(trainingData)
            val miniBatches = listOf(trainingData)
        }
    }

    fun updateMiniBatch(){

    }

    fun backprop(){

    }

    fun eval(){

    }

    fun costDeriv(){

    }



}