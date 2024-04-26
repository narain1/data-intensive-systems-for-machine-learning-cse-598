import breeze.linalg.DenseMatrix
import scala.util.Random

object MatrixMultiplyBenchmark {
    def main(args: Array[String]): Unit = {
        val n = if (args.length > 0) args(0).toInt else 1000
        val runs = if (args.length > 1) args(1).toInt else 10

        val matrixA = DenseMatrix.rand[Double](n, n)
        val matrixB = DenseMatrix.rand[Double](n, n)

        for (_ <- 0 until 5) {
            val _ =  matrixA * matrixB
        }

        val StartTime = System.nanoTime()
        for (_ <- 0 until runs) {
            val result = matrixA * matrixB
        }

        val endTime = System.nanoTime()

        val durationSeconds = (endTime - StartTime) / 1e9d
        println(s"Average time for $runs runs (Matrix size: $n x $n): ${durationSeconds / runs} seconds per run")
    }
}
