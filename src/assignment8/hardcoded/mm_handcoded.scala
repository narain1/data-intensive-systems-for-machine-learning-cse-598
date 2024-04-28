object MatrixMultiply {
  def main(args: Array[String]): Unit = {
    // Function to generate a matrix with random integer values
    def generateMatrix(rows: Int, cols: Int, maxValue: Int): Array[Array[Int]] = {
      Array.fill(rows, cols)(scala.util.Random.nextInt(maxValue))
    }

    // Function to multiply two matrices
    def multiplyMatrices(a: Array[Array[Int]], b: Array[Array[Int]]): Array[Array[Int]] = {
      require(a(0).length == b.length, "The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
      val result = Array.ofDim[Int](a.length, b(0).length)
      for (i <- result.indices; j <- result(0).indices) {
        result(i)(j) = (0 until a(0).length).map(k => a(i)(k) * b(k)(j)).sum
      }
      result
    }

    // Function to print the matrix
    def printMatrix(matrix: Array[Array[Int]]): Unit = {
      matrix.foreach { row =>
        println(row.mkString(" "))
      }
    }

    val rows = 1000
    val cols = 1000
    val maxValue = 100

    val matrix1 = generateMatrix(rows, cols, maxValue)
    val matrix2 = generateMatrix(cols, rows, maxValue) // cols of matrix1 should match rows of matrix2 for multiplication

    val startTime = System.nanoTime()
    val resultMatrix = multiplyMatrices(matrix1, matrix2)
    val endTime = System.nanoTime()

    println(s"Matrix multiplication took ${(endTime - startTime) / 1e9d} seconds.")

    // Optionally print one of the matrices
    // println("Matrix 1:")
    // printMatrix(matrix1)
    // println("Matrix 2:")
    // printMatrix(matrix2)
    // println("Result Matrix:")
    // printMatrix(resultMatrix)
  }
}

