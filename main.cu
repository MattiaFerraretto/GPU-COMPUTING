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

__global__ void cudaInit(ndarray A, float value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < A.shape[0] * A.shape[1])
        A.data[i] = value;
}   

__host__ ndarray* cuda_ndarray(int rows, int columns, float value)
{
    ndarray* A_dev = (ndarray*) malloc(sizeof(ndarray));
    int* shape = (int*) malloc(2 * sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    int n = rows * columns;
    
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->shape), 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->data), n * sizeof(float)));

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
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->data), n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy((void*)(A_dev->shape), (void*)(A->shape), 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(A_dev->data), (void*)(A->data), n * sizeof(float), cudaMemcpyHostToDevice));

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

    A->data = (float*) malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy((void*)(A->data), (void*)(A_dev->data), n * sizeof(float), cudaMemcpyDeviceToHost));

    if(!_free)
    {
        return A;
    }
    
    cudaFree_(A_dev);

    return A;
}

__global__ void cudaMTranspose(ndarray A, ndarray AT)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int rows, columns;
    __shared__ float tile[BLOCKDIMX][BLOCKDIMY];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        rows = A.shape[0];
        columns = A.shape[1];
    }
    __syncthreads();
    
    if(i >= rows || j >= columns)
        return ;

    tile[threadIdx.y][threadIdx.x] = A.data[columns * i + j];
    __syncthreads();
  
    AT.data[rows * j + i] = tile[threadIdx.y][threadIdx.x];
}

__host__ ndarray* cudaMTranspose(ndarray* A, bool A_devFree, bool AT_devFree)
{
    ndarray *A_dev = toDevice(A, false);
    ndarray *AT_dev = cuda_ndarray(A->shape[1], A->shape[0], -1);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (A->shape[0] + block.y - 1) / block.y;
    dim3 grid(gridx, gridy);

    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMTranspose <<<grid, block>>>(*A_dev, *AT_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    return toHost(AT_dev, AT_devFree);
}

__global__ void cudaMSub(ndarray A, ndarray B, ndarray C)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int rows, columns;
    __shared__ float A_tile[BLOCKDIMX][BLOCKDIMY];
    __shared__ float B_tile[BLOCKDIMX][BLOCKDIMY];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        rows = A.shape[0];
        columns = A.shape[1];
    }
    __syncthreads();

    if(i >= rows  || j >= columns)
        return ;

    A_tile[threadIdx.y][threadIdx.x] = A.data[columns * i + j];
    B_tile[threadIdx.y][threadIdx.x] = B.data[columns * i + j];

    __syncthreads();

    C.data[columns * i + j] = A_tile[threadIdx.y][threadIdx.x] - B_tile[threadIdx.y][threadIdx.x];
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
    ndarray* C_dev = cuda_ndarray(A->shape[0], A->shape[1], -1);

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

__global__ void cudaMSProduct(ndarray A, ndarray C, float value)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int rows, columns;
    __shared__ float tile[BLOCKDIMX][BLOCKDIMY];
    __shared__ float scalar;

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        rows = A.shape[0];
        columns = A.shape[1];
        scalar = value;
    }
    __syncthreads();

    if(i >=  rows || j >= columns)
        return ;

    tile[threadIdx.y][threadIdx.x] = A.data[columns * i + j];
    __syncthreads();

    C.data[columns * i + j] = tile[threadIdx.y][threadIdx.x] * scalar;
}

__host__ ndarray* cudaMSProduct(ndarray* A, double value, bool A_devFree, bool C_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* C_dev = cuda_ndarray(A->shape[0], A->shape[1], -1);

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

    __shared__ int n;
    __shared__ float A_tile[BLOCKDIMX * BLOCKDIMY];
    __shared__ float B_tile[BLOCKDIMX * BLOCKDIMY];

    if(threadIdx.x == 0)
        n = A.shape[1];
    
    __syncthreads();

    if(i >= n)
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

__host__ float cudaEDistance(ndarray* A, ndarray* B, bool A_devFree, bool B_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* B_dev = toDevice(B, false);
    
    dim3 block(BLOCKDIMX * BLOCKDIMY);
    dim3 grid((A->shape[1] + block.x - 1) / block.x);

    printf("grid: %d, block: %d\n", grid.x, block.x);

    ndarray* C_dev = cuda_ndarray(1, grid.x, -1);

    cudaEDistance <<<grid, block>>>(*A_dev, *B_dev, *C_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    ndarray* C = toHost(C_dev, true);

    double norm = 0;

    for(int i = 0; i < C->shape[1]; i++)
        norm += C->data[i];

    return sqrt(norm);
}

__global__ void cudaVSDivision(ndarray A, ndarray C, float value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int n;
    __shared__ float tile[BLOCKDIMX * BLOCKDIMY];
    __shared__ float scalar;
    
    if(threadIdx.x == 0)
    {
        n = A.shape[1];
        scalar = value;
    }
    __syncthreads();

    if(i >= n)
        return ;

    tile[tid] = A.data[i];
    __syncthreads();

    C.data[i] = tile[tid] / scalar; 
}

__host__ ndarray* cudaVSDivision(ndarray* A, float value, bool A_devFree, bool C_devFree)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* C_dev = cuda_ndarray(1, A->shape[1], -1);

    dim3 block(BLOCKDIMX * BLOCKDIMY);
    dim3 grid((A->shape[1] + block.x - 1) / block.x);

    cudaVSDivision <<< grid, block >>> (*A_dev, *C_dev, value);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(A_devFree)
        cudaFree_(A_dev);

    return toHost(C_dev, C_devFree);
}

__global__ void cudaMMProduct(ndarray A, ndarray B, ndarray C)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int aRws, aCls, bRws, bCls, cRws, cCls;
    __shared__ float A_tile[BLOCKDIMX][BLOCKDIMY];
    __shared__ float B_tile[BLOCKDIMX][BLOCKDIMY];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {   
        aRws = A.shape[0];
        aCls = A.shape[1];
        bRws = B.shape[0];
        bCls = B.shape[1];
        cRws = C.shape[0];
        cCls = C.shape[1];
    }
    __syncthreads();


    float sum = 0.f;
    for(int block = 0; block < gridDim.x; block++)
    {

        if(i <  aRws && j < aCls)
            A_tile[threadIdx.y][threadIdx.x] = A.data[aCls * i + blockDim.x * block + j];

        if(i < bRws && j < bCls)
            B_tile[threadIdx.y][threadIdx.x] = B.data[bCls * i + blockDim.y * bCls * block + j];
        
        __syncthreads();


        //TODO LOOP UNROLLING AND TUNING LAST BLOCK
        for(int k = 0; k < blockDim.x; k++)
        {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(i < cRws && j < cCls)
        C.data[cCls * i + j] = sum;
}

__host__ ndarray* cudaMMProduct(ndarray* A, ndarray* B)
{
    ndarray* A_dev = toDevice(A, false);
    ndarray* B_dev = toDevice(B, false);
    ndarray* C_dev = cuda_ndarray(A->shape[0], B->shape[1], 0);

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    int gridx = (A->shape[1] + block.x - 1) / block.x;
    int gridy = (B->shape[0] + block.y - 1) / block.y;
    dim3 grid(gridx, gridy);


    printf("grid: (%d, %d); blockSize: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaMMProduct <<< grid, block >>>(*A_dev, *B_dev, *C_dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree_(A_dev);
    cudaFree_(B_dev);

    return toHost(C_dev, true);
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

    float d[8] = {1, 2, 2, 1, 3, 4, 4, 3};
    ndarray* A = new_ndarray(4,2, d);
    //init(A);

    //ndarray* B = new_ndarray(5,5, NULL);
    //init(B);

    ndarray* AT = cudaMTranspose(A, true, true);
   

    //ndarray* C = cudaMSub(A, A, true, true, true);

    //ndarray* C = cudaMatScalarProduct(A, 2, true, true);

    //double nrm = cudaEDistance(A, B, true, true);
    //printf("%2f\n", nrm);

    //ndarray* C = cudaVSDivision(A, 2, true, true);
    //print(A);

    ndarray* C = cudaMMProduct(AT, A);
    
    print(A);
    print(AT);
    print(C);

    

}