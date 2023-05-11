#include <stdio.h>
#include <time.h>

#include "clib/ndarray.h"


#define CUDA_CHECK(call)                                                        \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("ERROR:: File: %s, Line: %d, ", __FILE__, __LINE__);             \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

#define BLOCKDIMX 16
#define BLOCKDIMY 16

__global__ void cudaInit(ndarray A, double value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < A.shape[0] * A.shape[1])
        A.data[i] = value;
}   

__host__ ndarray* cudaNewndarray(int rows, int columns, double value)
{
    ndarray* A_dev = (ndarray*) malloc(sizeof(ndarray));
    int* shape = (int*) malloc(2 * sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    int n = rows * columns;
    
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->shape), 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->data), n * sizeof(double)));

    //CUDA_CHECK(cudaMemset((void*)(A_dev->data), *((long*)&value), n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy((void*)(A_dev->shape), (void*)(shape), 2 * sizeof(int), cudaMemcpyHostToDevice));
    free(shape);
    
    dim3 block(BLOCKDIMX * BLOCKDIMY);
    dim3 grid((rows * columns + block.x - 1) / block.x);
   
    //printf("%d, %d\n", grid.x, block.x);

    cudaInit <<<grid, block>>> (*A_dev, value);
    CUDA_CHECK(cudaDeviceSynchronize());    

    return A_dev;
}

__host__ void cudaFree_(ndarray* A_dev)
{
    cudaFree(A_dev->shape);
    cudaFree(A_dev->data);
    free(A_dev);
}

__host__ ndarray* toDevice(ndarray* A, bool _free)
{
    ndarray *A_dev = (ndarray*) malloc(sizeof(ndarray));
    int n =  A->shape[0] * A->shape[1];

    CUDA_CHECK(cudaMalloc((void**)&(A_dev->shape), 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->data), n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy((void*)(A_dev->shape), (void*)(A->shape), 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(A_dev->data), (void*)(A->data), n * sizeof(double), cudaMemcpyHostToDevice));

    if(!_free)
    {
        return A_dev;
    }  
    
    free_(A);

    return A_dev;
}

__host__ ndarray* toHost(ndarray* A_dev, bool _free)
{
    ndarray* A = (ndarray*) malloc(sizeof(ndarray));

    A->shape = (int*) malloc(2 * sizeof(int));
    CUDA_CHECK(cudaMemcpy((void*)(A->shape), (void*)(A_dev->shape),  2 * sizeof(int), cudaMemcpyDeviceToHost));
    
    int n = A->shape[0] * A->shape[1];

    A->data = (double*) malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy((void*)(A->data), (void*)(A_dev->data), n * sizeof(double), cudaMemcpyDeviceToHost));

    if(!_free)
    {
        return A;
    }
    
    cudaFree_(A_dev);

    return A;
}

__global__ void cudaTranspose(ndarray A, ndarray AT)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double tile[BLOCKDIMX][BLOCKDIMY];

    if(i >= A.shape[0] || j >= A.shape[1])
        return ;

    tile[threadIdx.y][threadIdx.x] = A.data[A.shape[1] * i + j];
    __syncthreads();
  
     AT.data[A.shape[0] * j + i] = tile[threadIdx.y][threadIdx.x];
}

__host__ ndarray* cudaTranspose(ndarray* A, bool _free)
{
    ndarray *A_dev = toDevice(A, false);
    ndarray *AT_dev = cudaNewndarray(A->shape[1], A->shape[0], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.y - 1) / block.y;

    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaTranspose <<<grid, block>>>(*A_dev, *AT_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    return toHost(AT_dev, _free);
}

__global__ void cudaMatSub(ndarray A, ndarray B, ndarray C)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double A_tile[BLOCKDIMX][BLOCKDIMY];
    __shared__ double B_tile[BLOCKDIMX][BLOCKDIMY];

    if(i >= A.shape[0]  || j >= A.shape[1])
        return ;

    A_tile[threadIdx.y][threadIdx.x] = A.data[A.shape[1] * i + j];
    B_tile[threadIdx.y][threadIdx.x] = B.data[A.shape[1] * i + j];

    __syncthreads();

    C.data[A.shape[1] * i + j] = A_tile[threadIdx.y][threadIdx.x] - B_tile[threadIdx.y][threadIdx.x];
}

__host__ ndarray* cudaMatSub(ndarray* A, ndarray* B, bool _free)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* B_dev = toDevice(B, false);
    ndarray* C_dev = cudaNewndarray(A->shape[0], A->shape[1], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.x - 1) / block.y;

    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMatSub <<<grid, block>>>(*A_dev, *B_dev, *C_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    return toHost(C_dev, _free);
}

void init(ndarray* A)
{
    srand(time(NULL));

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        A->data[i] = (double)rand() / (double)RAND_MAX;
    }
}

int main(){

    ndarray* A = new_ndarray(20,30, NULL);
    init(A);

    ndarray* B = new_ndarray(2,2, NULL);
    init(B);

    //ndarray* AT = cudaTranspose(A, true);
   

    ndarray* C = cudaMatSub(A, A, true);
    
    print(A);
    print(C);
    //print(AT);

}