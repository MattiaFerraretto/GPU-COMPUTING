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

ndarray* toDevice(ndarray* A, bool _free)
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

ndarray* toHost(ndarray* A_dev, bool _free)
{
    ndarray *A = (ndarray*) malloc(sizeof(ndarray));

    A->shape = (int*) malloc(2 * sizeof(int));
    CUDA_CHECK(cudaMemcpy((void*)(A->shape), (void*)(A_dev->shape),  2 * sizeof(int), cudaMemcpyDeviceToHost));
    
    int n = A->shape[0] * A->shape[1];

    A->data = (double*) malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy((void*)(A->data), (void*)(A_dev->data), n * sizeof(double), cudaMemcpyDeviceToHost));

    if(!_free)
    {
        return A;
    }
    
    cudaFree(A_dev->shape);
    cudaFree(A_dev->data);
    free(A_dev);

    return A;
}

__global__ void init(ndarray A)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    A.data[i] = i;

}



__global__ void cudaTranspose(ndarray A)
{
    int tidx = blockDim.y * threadIdx.x + threadIdx.y;
    int tidy = blockDim.x * threadIdx.y + threadIdx.x;

    int i = gridDim.x * blockIdx.x + tidx;

    //__shared__ double sm[4];
    


    printf("bid: %d, tidx: %d, tidy: %d, i: %d\n", blockIdx.x, tidx, tidy, i);

    //printf("i: %d * %d + %d = %d\n", blockDim.x, blockIdx.x, threadIdx.x, i);
    //printf("gmi: %d + %d = %d\n", blockIdx.x, smi, blockIdx.x + smi);
    //printf("j: %d * %d + %d\n", blockDim.y, blockIdx.y, threadIdx.y);
    //printf("tid:%d, %d\nsmi: %d\ngmi: %d\n", threadIdx.x, threadIdx.y, smi, gmi);



}

int main(){

    
    ndarray *A = new_ndarray(4, 4, NULL);
    int n = A->shape[0] * A->shape[1];

    ndarray* A_dev = toDevice(A, true);

    init <<<1, n>>> (*A_dev);
    cudaDeviceSynchronize();

    print(toHost(A_dev, false));


    

    cudaTranspose <<<4, dim3(2,2)>>>(*A_dev);
    cudaDeviceSynchronize();

    //a = toHost(adev, true);
    //print(a);

}