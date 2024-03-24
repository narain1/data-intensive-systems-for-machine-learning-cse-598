#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}


__global__ void ArraySetKernel(int numElements, float *arr, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    arr[idx] = value;
  }
}

__global__ void BroadcastToKernel(int in_threads, int out_threads, const float *input, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < out_threads) {
    output[idx] = input[idx % in_threads];
  }
}

__global__ void ReduceSumAxisZero(const float *input_data, float *output_data, int rows, int input) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < input) {
    output_data[idx] = 0.0f;
    for (int i=0; i<rows/input; i++) {
      output_data[idx] += input_data[i*input+idx];
    }
  }
}

__global__ void MatrixElementWiseAddByConst(int numElements, const float *a, float val, float *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    c[idx] = a[idx] + val;
  }
}


__global__ void MatrixElementWiseAdd(int numElements, const float *a, const float *b, float *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    c[idx] = a[idx] + b[idx];
  }
}


__global__ void MatrixElementWiseMultiply(int numElements, const float *a, const float *b, float *o) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    o[idx] = a[idx] * b[idx];
  }
}


__global__ void MatrixElementWiseMultiplyConst(int numElements, const float *a, float val, float *o) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    o[idx] = a[idx] * val;
  }
}


__global__ void reluKernel(int numElements, const float *a, float *o) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        o[idx] = max(0.0f, a[idx]);
    }
}

__global__ void ReluGradient(int numElements, const float *a, const float *g, float *o) {
  int idx = blockIdx.x + blockDim.x + threadIdx.x;
  if (idx < numElements) {
    o[idx] = (a[idx] > 0) ? g[idx] : 0;
  }
}


__global__ void Softmax(const float *input, float *output, int r, int c) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < r) {
    float max_val = -FLT_MAX;
    for (int i=0; i<c; i++) {
      max_val = fmaxf(max_val, input[row * c + i]);
    }

    float sum = 0.0f;
    for (int i=0; i<c; i++) {
      sum += expf(input[row * c + i] - max_val);
    }

    for (int i=0; i<c; i++) {
      output[row*c+i] = expf(input[row*c+i] - max_val) / sum;
    }
  } 
}
    



int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int threads_per_block = 512;
  index_t number_of_threads = 1;
  for (int i=0; i<arr->ndim; i++) {
    number_of_threads = number_of_threads * arr->shape[i];
  }
  float *data = (float *)arr->data;
  int blocks = (n + threads_per_block -1) / threads_per_block;
  ArraySetKernel<<<blocks, threads_per_block>>>(number_of_threads, data, value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int in_threads = 1;
  int out_threads = 1;
  int threads_per_block = 512;
  for (int i=0; i<input->ndim; i++) {
    in_threads = in_threads * input->shape[i];
  }
  for (int i=0; i<output->ndim; i++) {
    out_threads = out_threads * output->shape[i];
  }
  const float *input_data = (const float*)input->data;
  float *output_data = (float*)output->data;
  int n_blocks = (out_threads + threads_per_block - 1) / threads_per_block;
  BroadcastToKernel<<<n_blocks, threads_per_block>>>(in_threads, out_threads, input_data, output_data);
  return 0;
}


int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int threads_per_block = 512;

  index_t input_n = 1, output_n = 1;
  for (int i=0; i<input->ndim; i++) {
    input_n *= input->shape[i]
  }

  for (int i=0; i<output->ndim; i++) {
    output_n *= output->shape[i];
  }

  const float *input_data = (const float*)input->data;
  float *output_data = (float*)output->data;
  int n_blocks = (output_n + threads_per_block - 1) / threads_per_block;
  ReduceSumAxisZero<<<n_blocks, threads_per_block>>>(input_data, output_data, input_n, output_n);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<output->ndim; i++) {
    n *= output->shape[i];
  }

  const float *matA_data = (const float*)matA->data;
  const float *matB_data = (const float*)matB->data;
  float *output_data = (float*)output->data;

  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;
  MatrixElementWiseAdd<<<n_blocks, threads_per_block>>>(number_of_threads, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<output->ndim; i++)
    n *= output->shape[i];
  
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1)/threads_per_block;
  MatrixElementWiseAddByConst<<<n_blocks, threads_per_block>>>(number_of_threads, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<output->ndim; i++)
    n *= output->shape[i];
  
  const float *matA_data = (const float*)matA->data;
  const float *matB_data = (const float*)matB->data;
  float *output_data = (float*)output->data;
  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;
  MatrixElementWiseMultiply<<<n_blocks, threads_per_block>>>(n, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<output->ndim; i++)
    n *= output->shape[i];
  
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;

  MatrixElementWiseMultiplyConst<<<n_blocks, threads_per_block>>>(n, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreat e(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    printf("CUBLAS initialization failed\n");

  const float *matA_data = (const float*)matA->data;
  const float *matB_data = (const float*)matB->data;
  float *matC_data = (float *)matC->data;

  cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = trasnposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int m = transposeB ?  matB->shape[0] : matB->shape[1];
  int n = transposeA ? matA->shape[1] : matA->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  float alpha = 1.0f;
  float beta = 0.0f;
  stat = cublasSgemm(handle, transb, transa,
                    m, n, k, &alpha, matB_data,
                    matB->shape[1], matA_data, matA->shape[1],
                    &beta, matC_data, m);

  if (stat != CUBLAS_STATUS_SUCCESS) 
    printf("CUBLAS kernel execution error");

  stat = cublasDestroy(handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    printf("CUBLAS shutdown error\n");
  return 0;
}



int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<output->ndim; i++)
    n *= output->shape[i];
  
  const float *input_data = (const float*)input->data;
  float *output_data = (float *)output->data;
  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;
  reluKernel<<<n_blocks, threads_per_block>>>(n, input_data, output_data);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i=0; i<input->ndim; i++)
    n *= input->shape[i];
  
  const float *input_data = (const float*)input->data;
  const float *in_grad_data = (const float*)in_grad->data;
  float *output_data = (float*)output->data;

  int threads_per_block = 512;
  int n_blocks = (n + threads_per_block - 1) / threads_per_block;
  ReluGradient<<<n_blocks, threads_per_block>>>(n, input_data, in_grad_data, output_data);
  return 0;
}


int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  const float *input_data = (const float*)input->data;
  float *output_data = (float*)output->data;
  int n_blocks = output->shape[0];

  Softmax<<<n_blocks, 1>>>(input_data, output_data, output->shape[0], output->shape[1]);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
