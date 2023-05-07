#include<stdio.h>

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

__host__ void cudaFree_(ndarray* A_dev)
{
    cudaFree(A_dev->shape);
    cudaFree(A_dev->data);
    free(A_dev);
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
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int rIndex = blockDim.x * blockDim.y * i + j;
    int cIndex = blockDim.x * blockDim.y * j + i;

    //int rIndex = A.shape[0] * A.shape[1] * i + j;
    //int cIndex = A.shape[0] * A.shape[1] * j + i;


    __shared__ double sm[16][16];

    if(rIndex >  A.shape[0] * A.shape[1]){
       
        return ;

    }
    
    printf("(%d, %d) <- %d, %d; (%d, %d); (%d, %d)\n", threadIdx.x,  threadIdx.y, rIndex, cIndex, i, j, A.shape[0], A.shape[1]);

    sm[threadIdx.x][threadIdx.y] = A.data[rIndex];
    __syncthreads();

    AT.data[cIndex] = sm[threadIdx.x][threadIdx.y];
}

__host__ ndarray* cudaTranspose(ndarray* A)
{
    ndarray *AT = new_ndarray(A->shape[1], A->shape[0], NULL);

    ndarray *A_dev = toDevice(A, false);
    ndarray *AT_dev = toDevice(AT, true);

    dim3 blockSize(16, 16);

    int gridx = (A->shape[1] + blockSize.x - 1) / blockSize.x;
    int gridy = (A->shape[0] + blockSize.x - 1) / blockSize.y;

    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, blockSize.x, blockSize.y);

    cudaTranspose <<<grid, blockSize>>>(*A_dev, *AT_dev);
    cudaDeviceSynchronize();

    return toHost(AT_dev, true);
}

void init(ndarray* A)
{
    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        A->data[i] = i;
    }
}

int main(){

    
    ndarray* A = new_ndarray(4, 4, NULL);
    init(A);

    ndarray* AT = cudaTranspose(A);
    
    print(A);
    print(AT);

}