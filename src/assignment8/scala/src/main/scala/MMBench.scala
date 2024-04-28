import breeze.linalg.DenseMatrix
import scala.util.Random

object MatrixMultiplyBenchmark {
    def main(args: Array[String]): Unit = {
        val n = if (args.length > 0) args(0).toInt else 1000
        val runs = if (args.length > 1) args(1).toInt else 1000

        val matrixA = DenseMatrix.rand[Double](n, n)
        val matrixB = DenseMatrix.rand[Double](n, n)

        // Warm-up phase to help JVM optimization
        var dummy = 0.0
        for (_ <- 0 until 5) {
            dummy += (matrixA * matrixB).apply(0, 0) // Use a small part of the result to prevent discarding the computation
        }
        println(s"Warm-up phase complete, dummy value: $dummy") // This can be removed or commented out

        val startTime = System.nanoTime()
        for (_ <- 0 until runs) {
            val result = matrixA * matrixB
        }

        val endTime = System.nanoTime()

        val durationSeconds = (endTime - startTime) / 1e9d
        println(s"Average time for $runs runs (Matrix size: $n x $n): ${durationSeconds} seconds")
    }
}

