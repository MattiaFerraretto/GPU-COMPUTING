#include<stdio.h>

#include "clib/ndarray.h"


#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("ERROR:: File: %s, Line: %d, ", __FILE__, __LINE__);             \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}


__global__ void hello(ndarray* A){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Thread %d: %2f\n", i, A->data[i]);
}

ndarray* toDevice(ndarray* A)
{

    //ndarray *A_dev = (ndarray*) malloc(sizeof(ndarray));
    ndarray* A_dev;
    CHECK(cudaMalloc((void**)&A_dev, sizeof(ndarray)));

    CHECK(cudaMalloc((void**)&(A_dev->shape), 2 * sizeof(int)));
    CHECK(cudaMalloc((void**)&(A_dev->data), A->shape[0] * A->shape[1] * sizeof(double)));


    CHECK(cudaMemcpy((void*)A_dev->shape, (void*)A->shape, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)A_dev->data, (void*)A->data, A->shape[0] * A->shape[1] * sizeof(double), cudaMemcpyHostToDevice));

    return A_dev;
}




int* toDevice1(int* A, int n){

    int* a;
    cudaMalloc((void**)&a, n * sizeof(int));


    
    cudaMemcpy((void*)a, (void*)A, n * sizeof(int), cudaMemcpyHostToDevice);

    return a;
} 


void init(double* V, int n)
{
    for(int i = 0; i < n; i++)
        V[i] = 1;
}

int main(){

    int n = 20;
    

    ndarray* a = new_ndarray(2, 10, NULL);
    init(a->data, n);
    print(a);
    //fun(2);

    ndarray* adev = toDevice(a);

    //int* A_dev = toDevice1(v, n);
    //int* B_dev = toDevice1(v, n);

    hello <<<1, n>>>(adev);
    cudaDeviceSynchronize();

}