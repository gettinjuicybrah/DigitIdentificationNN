/* =======================================================================================
   MAIN FUNCTION
   ======================================================================================= */

/**
 * Main function demonstrating training and testing of the SimpleCNN on MNIST data.
 */
fun main() {
    val cnn = SimpleCNN()
    val learningRate = 0.01
    val epochs = 5

    // Load training data from CSV.
    val trainingData = loadMNISTData("mnist_train.csv")
    println("Training started...")
    for (epoch in 1..epochs) {
        var totalLoss = 0.0
        trainingData.shuffled().forEach { (image, target) ->
            totalLoss += cnn.trainExample(image, target, learningRate)
        }
        println("Epoch $epoch, Loss: ${totalLoss / trainingData.size}")
    }
    println("Training finished.")

    // Save the model.
    cnn.saveModel("cnn_model.txt")
    println("Model saved to cnn_model.txt")

    // Load test data from CSV.
    val testData = loadMNISTData("mnist_test.csv")
    var correct = 0
    for ((image, target) in testData) {
        val prediction = cnn.predict(image)
        val targetLabel = target.indices.indexOfFirst { target[it] == 1.0 }
        if (prediction == targetLabel) correct++
    }
    println("Test accuracy: ${correct.toDouble() / testData.size}")
}