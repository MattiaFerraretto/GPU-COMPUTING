#include <stdio.h>
#include <time.h>
#include <math.h>

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
    CUDA_CHECK(cudaFree(A_dev->shape));
    CUDA_CHECK(cudaFree(A_dev->data));
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

__global__ void cudaMTransope(ndarray A, ndarray AT)
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

__host__ ndarray* cudaMTransope(ndarray* A, bool A_devFree, bool AT_devFree)
{
    ndarray *A_dev = toDevice(A, false);
    ndarray *AT_dev = cudaNewndarray(A->shape[1], A->shape[0], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.y - 1) / block.y;
    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMTransope <<<grid, block>>>(*A_dev, *AT_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    return toHost(AT_dev, AT_devFree);
}

__global__ void cudaMSub(ndarray A, ndarray B, ndarray C)
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

__host__ ndarray* cudaMSub(ndarray* A, ndarray* B, bool A_devFree, bool B_devFree, bool C_devFree)
{
    if(A->shape[0] != A->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* A_dev = toDevice(A, false);
    ndarray* B_dev = toDevice(B, false);
    ndarray* C_dev = cudaNewndarray(A->shape[0], A->shape[1], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.x - 1) / block.y;
    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMSub <<<grid, block>>>(*A_dev, *B_dev, *C_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    if(B_devFree)
        cudaFree_(B_dev);

    return toHost(C_dev, C_devFree);
}

__global__ void cudaMSProduct(ndarray A, ndarray C, double value)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double tile[BLOCKDIMX][BLOCKDIMY];

    if(i >=  A.shape[0] || j >= A.shape[1])
        return ;

    tile[threadIdx.y][threadIdx.x] = A.data[A.shape[1] * i + j];
    __syncthreads();

    C.data[A.shape[1] * i + j] = tile[threadIdx.y][threadIdx.x] * value;
}

__host__ ndarray* cudaMSProduct(ndarray* A, double value, bool A_devFree, bool C_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* C_dev = cudaNewndarray(A->shape[0], A->shape[1], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.x - 1) / block.y;
    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMSProduct <<<grid, block>>>(*A_dev, *C_dev, value);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    return toHost(C_dev, C_devFree);
}

__global__ void cudaEDistance(ndarray A, ndarray B, ndarray C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ double A_tile[BLOCKDIMX * BLOCKDIMY];
    __shared__ double B_tile[BLOCKDIMX * BLOCKDIMY];
    //__shared__ double C_tile[BLOCKDIMX * BLOCKDIMY];

    if(i >= A.shape[1])
        return ;

    A_tile[tid] = A.data[i];
    B_tile[tid] = B.data[i];
    __syncthreads();

    A_tile[tid] = pow(A_tile[tid], 2);
    B_tile[tid] = pow(B_tile[tid], 2);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            A_tile[tid] = (A_tile[tid] - B_tile[tid]) + (A_tile[tid + stride] - B_tile[tid + stride]);
        }
        __syncthreads();
    }

    if(tid == 0)
        C.data[blockIdx.x] = A_tile[0];

}

__host__ double cudaEDistance(ndarray* A, ndarray* B, bool A_devFree, bool B_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* B_dev = toDevice(B, false);
    
    dim3 block(BLOCKDIMX * BLOCKDIMY);
    dim3 grid((A->shape[1] + block.x - 1) / block.x);

    printf("grid: %d, block: %d\n", grid.x, block.x);

    ndarray* C_dev = cudaNewndarray(1, grid.x, -1);

    cudaEDistance <<<grid, block>>>(*A_dev, *B_dev, *C_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    ndarray* C = toHost(C_dev, true);

    double norm = 0;

    for(int i = 0; i < C->shape[1]; i++)
        norm += C->data[i];

    return sqrt(norm);
}

__global__ void cudaVSDivision(ndarray A, ndarray C, double value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ double tile[BLOCKDIMX * BLOCKDIMY];

    if(i >= A.shape[1])
        return ;

    tile[tid] = A.data[i];
    __syncthreads();

    C.data[i] = tile[tid] / value; 
}

__host__ ndarray* cudaVSDivision(ndarray* A, double value, bool A_devFree, bool C_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* C_dev = cudaNewndarray(1, A->shape[1], -1);

    dim3 block(BLOCKDIMX * BLOCKDIMY);
    dim3 grid((A->shape[1] + block.x - 1) / block.x);

    cudaVSDivision <<< grid, block >>> (*A_dev, *C_dev, value);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    return toHost(C_dev, C_devFree);
}

void init(ndarray* A)
{
    srand(time(NULL));

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        //A->data[i] = (double)rand() / (double)RAND_MAX;
        A->data[i] = 1;
    }
}

int main(){

    ndarray* A = new_ndarray(1,20, NULL);
    init(A);

    ndarray* B = new_ndarray(1,20, NULL);
    //init(B);

    //ndarray* AT = cudaTranspose(A, true);
   

    //ndarray* C = cudaMatSub(A, A, true, true, true);

    //ndarray* C = cudaMatScalarProduct(A, 2, true, true);

    double nrm = cudaEDistance(A, B, true, true);
    printf("%2f\n", nrm);

    ndarray* C = cudaVSDivision(A, nrm, true, true);
    print(A);
    print(C);
    //print(AT);

}