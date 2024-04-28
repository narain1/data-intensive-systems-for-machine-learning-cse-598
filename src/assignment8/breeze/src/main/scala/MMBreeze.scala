import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.exp

object MMBreeze{
    def main(args: Array[String]) {
        val runs = 1000
        val n = 1000
        val arr1 = DenseMatrix.rand[Double](1000, 1000)
        val arr2 = DenseMatrix.rand[Double](1000, 1000)

        var result = arr1 * arr2

        val start = System.currentTimeMillis()
        for (_ <- 1 to runs) 
            result = arr1 * arr2
        val end = System.currentTimeMillis()

        println(s"\n Time for $runs runs: ${(end - start) / 1000.0} s\n")

    }
}