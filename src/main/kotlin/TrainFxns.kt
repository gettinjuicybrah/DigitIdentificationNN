
fun sigmoid(input: Double): Double{
    return 1.0/(1.0+kotlin.math.exp(-input))
}

fun sigmoidPrime(input: Double): Double{
    return sigmoid(input) * (1 - sigmoid(input))
}

fun dotProduct(v1: List<Double>, v2: List<Double>): Double{

    return v1.zip(v2){
        x, y -> x * y
    }.sum()

}