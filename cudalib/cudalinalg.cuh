#ifndef cudalinalg_h
#define cudalinalg_h

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <cstring>


#include "../clib/ndarray.h"

/**
 * Macro to print cuda error
*/
#define CUDA_CHECK(call)                                                        \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("CERROR:: File: %s, Line: %d, ", __FILE__, __LINE__);            \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

/**
 * Definition of block size
*/
#define BLOCKDIMX 16
#define BLOCKDIMY 16


/**
 * cuda_ndarrayHost fucntion allocates a n-dimensional array (rows x columns). 
 * The difference with respect to the function implemented in ndarry library (new_ndarray) is that,
 * in order to improve the performace of device's writing and reading operations it was used the page-loked or pinned memory for data array.
 * So be carefully, because pinned memory can degradates host performance.
 * 
 * Parameters
 * ----------
 * rows     : number of rows
 * columns  : number of columns
 * 
 * Returns
 * -------
 * A new instance of a n-dimensional array (rows x columns).
 * 
*/
__host__ ndarray* cuda_ndarrayHost(int rows, int columns);

/**
 * cudaFreeHost_ function deallocates the n-dimensional array pointed to by the pointer.
 * This function is used to free the page-loked memory allocated with previus function.
 * 
 * Paramenters
 * -----------
 * A    : pointer to the n-dimensional array allocated with the pinned memory
 * 
*/
__host__ void cudaFreeHost_(ndarray* A);


/**
 * cuda_ndarry function directly allocates a new n-dimensional array (rows x columns) in the device's global memory.
 * The shape is immediatly copied to the device.
 * 
 * Parameters
 * ----------
 * rows     : number or rows
 * columns  : number of columns
 * 
 * Returns
 * -------
 * A new instace of a n-dimensional array in the device's global memory
 * 
*/
__host__ ndarray* cuda_ndarray(int rows, int columns);

/**
 * cudaFree_ function deallocates the n-dimensional array pointed to by the pointer in the device's global memory.
 * Any attemps to free a n-dimensional array allocated in the host memory causes the software to crash.
 * 
 * Parameters
 * ----------
 * A_dev    : pointer to a n-dimensional array allocated in the device's global memory.
 * 
*/
__host__ void cudaFree_(ndarray* A_dev);

/**
 * memTileH2DAsync is a primitive function used to transfer a memory tile from the host to the device in asynchronously.
 * This function should be used when you want to transfer a 1-dimensional memory tile. Any attempt to use it with n-dimensional memory tiles
 * may result in undefined behavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in device's global memory
 * src      : pointer to the source n-dimensional array allocated in the host memory
 * offset   : starting point for memory transfer (i-th element of 1-dimensional array)
 * dimTile  : dimension of memory tile (not in byte)
 * stream   : cuda stream on which the memory tile transfer is performed
 * 
*/
__host__ void memTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);

/**
 * memTileD2HAsync is a primitive function used to transfer a memory tile from the device to the host in asynchronously.
 * This function should be used when you want to transfer a 1-dimensional memory tile. Any attempt to use it with n-dimensional memory tiles
 * may result in undefined behavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in the host memory
 * src      : pointer to the source n-dimensional array allocated in device's global memory
 * offset   : starting point for memory transfer (i-th element of 1-dimensional array)
 * dimTile  : dimension of memory tile (not in byte)
 * stream   : cuda stream on which the memory tile transfer is performed
 * 
*/
__host__ void memTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);

/**
 * rowTileH2DAsync is a primitive function used to transfer 2-dimensional memory tile from host memory to device memory in asynchronously.
 * This function must be used with matrices. The 2-dimensional memory tile corresponds to the sub-matrix (dimTile x columns).
 * Any attempt to use it with a different dimensional memory tile may result in undefined bheavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in the device's global memory
 * src      : pointer to the source n-dimensional array allocated in the host memory
 * offset   : starting point for memory transfer (i-th row)
 * dimTile  : dimension of memory tile (number of rows to be transfered)
 * stream   : cuda stream on which the memory tile transfer is performed
*/
__host__ void rowTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);

/**
 * rowTileD2HAsync is a primitive function used to transfer 2-dimensional memory tile from device memory to host memory in asynchronously.
 * This function must be used with matrices. The 2-dimensional memory tile corresponds to the sub-matrix (dimTile x columns).
 * Any attempt to use it with a different dimensional memory tile may result in undefined bheavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in the host memory
 * src      : pointer to the source n-dimensional array allocated in the device's global memory
 * offset   : starting point for memory transfer (i-th row)
 * dimTile  : dimension of memory tile (number of rows to be transfered)
 * stream   : cuda stream on which the memory tile transfer is performed
*/
__host__ void rowTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);

/**
 * columnTileH2DAsync is a primitive function used to transfer 2-dimensional memory tile from host memory to device memory in asynchronously.
 * The function must be used with matrices. The 2-dimensional memory tile corresponds to the sub-matrix (rows x dimTile).
 * Any attempt to use it with a different dimensional memory tile may result in undefined bheavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in the device's global memory 
 * src      : pointer to the source n-dimensional array allocated in the host memory
 * offset   : starting point for memory transfer (i-th column)
 * dimTile  : dimension of memory tile (number of columns to be transfered)
 * stream   : cuda stream on which the memory tile transfer is performed
*/
__host__ void columnTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);

/**
 * columnTileH2DAsync is a primitive function used to transfer 2-dimensional memory tile from device to host memory in asynchronously.
 * The function must be used with matrices. The 2-dimensional memory tile corresponds to the sub-matrix (rows x dimTile).
 * Any attempt to use it with a different dimensional memory tile may result in undefined bheavior.
 * 
 * Parameters
 * ----------
 * dest     : pointer to the target n-dimensional array allocated in the host memory 
 * src      : pointer to the source n-dimensional array allocated in the device's global memory
 * offset   : starting point for memory transfer (i-th column)
 * dimTile  : dimension of memory tile (number of columns to be transfered)
 * stream   : cuda stream on which the memory tile transfer is performed
*/
__host__ void columnTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream);



/**
 * cudaVSDivision kernel performs the element-wise division between a vector and a scalar value.
 * Let A be a vector and let v be a scalar value, C = A / v
 * The kernel supports grid-level parallelism (by means of cuda streams) on inputs.
 * 
 * Parameters
 * ----------
 * A        : 1-dimensional array of the device
 * C        : 1-dimensional array of the device in which the result is written
 * value    : scalar value
 * offset   : starting point for kernel's operations (i-th array element)
 * dimTile  : Number of array elements to work on
*/
__global__ void cudaVSDivision(ndarray A, ndarray C, float value, int offset, int dimTile);

/**
 * cudaVSDivision host function is responsible for properly transfering memory and launching cudaVSDivision kernel.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 1-dimensional array
 * value    : scalar value
 * nTile    : Number of memory tiles
 * inPlace  : if true the result is written in A otherwise a new 1-dimensional array is returned
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the pointer to the host 1-dimensional array containing the element-wise division.
*/
__host__ ndarray* cudaVSDivision(ndarray* A, float value, int nTile, bool inPlace, bool verbose);


/**
 * cudaEDistance kernel performs the eucledian distance between two 1-dimensional vectors.
 * In the case one of them is the origin vector, the result is the Frobenius norm.
 * Let A and B be two vectors, C = sum( (A_i - B_i)^2 ), i = 1, ... , n
 * The kernel supports grid-level parallelism (by means of cuda streams) on inputs.
 * 
 * Parameters
 * ----------
 * A        : 1-dimensional array of the device
 * B        : 1-dimensional array of the device
 * C        : 1-dimensional array of the device in which the partial sums are written
 * offset   : starting point for kernel's operations (i-th array element)
 * dimTile  : Number of array elements to work on
 * coffset  : Starting point for kernel writes operations
*/
__global__ void cudaEDistance(ndarray A, ndarray B, ndarray C, int offset, int dimTile, int coffset);

/**
 * cudaVSDivision host function is responsible for properly transfering memory, launching cudaVSDivision kernel, and
 * summing partial results.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 1-dimensional array
 * B        : pointer to the host 1-dimensional array
 * nTiles   : Number of memory tiles
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the euclidean distance between two vectors.
*/
__host__ float cudaEDistance(ndarray* A, ndarray* B, int nTile, bool verbose);


/**
 * cudaMSProduct kernel performs the element-wise product between a matrix and a scalar value.
 * Let A be a matrix and let v be a scalar value, C = A * v
 * The kernel supports grid-level parallelism (by means of cuda streams) on inputs.
 * 
 * Parameters
 * ----------
 * A            : 2-dimensional array of the device
 * C            : 2-dimensional array of the device in which the result is written
 * value        : scalar value
 * rowOffset    : starting point for kernel's operations (i-th row)
 * rowsTile     : number of rows of the matrix to work on
*/
__global__ void cudaMSProduct(ndarray A, ndarray C, float value, int rowOffset, int rowsTile);

/**
 * cudaMSProduct host function is responsible for properly transfering memory and launching cudaMSProduct kernel.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 2-dimensional array
 * value    : scalar value
 * nTile    : Number of memory tiles
 * inPlace  : if true the result is written in A otherwise a new 2-dimensional array is returned
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the pointer to the host 2-dimensional array containing the element-wise product.
*/
__host__ ndarray* cudaMSProduct(ndarray* A, float value, int nTile, bool inPlace, bool verbose);


/**
 * cudaMTranspose kernel performs the transpose of a matrix.
 * The kernel supports grid-level parallelism (by means of cuda streams) on inputs.
 * 
 * Parameters
 * ----------
 * A            : 2-dimensional array of the device
 * AT           : transposed 2-dimensional array of the device 
 * rowOffset    : starting point for kernel's operations (i-th row)
 * rowsTile     : number of rows of the matrix to work on 
*/
__global__ void cudaMTranspose(ndarray A, ndarray AT, int rowOffset, int rowsTile);

/**
 * cudaMTranspose host function is responsible for properly transfering memory and launching cudaMTranspose kernel.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 2-dimensional array
 * nTile    : Number of memory tiles
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the pointer to the host 2-dimensional array containing the trasposed matrix.
*/
__host__ ndarray* cudaMTranspose(ndarray* A, int nTile, bool verbose);


/**
 * cudaMMSub kernel performs the subtraction between two matrices.
 * Let A, B, C be three matrices, C = A - B  
 * The kernel supports grid-level parallelism (by means of cuda streams) on inputs.
 * 
 * Parameters
 * ----------
 * A            : 2-dimensional array of the device
 * B            : 2-dimensional array of the device
 * C            : 2-dimensional array of the device in which the result is written
 * rowOffset    : starting point for kernel's operations (i-th row)
 * rowsTile     : number of rows of the matrix to work on
*/
__global__ void cudaMMSub(ndarray A, ndarray B, ndarray C, int rowOffset, int rowsTile);

/**
 * cudaMMSub host function is responsible for properly transfering memory and launching cudaMMSub kernel.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 2-dimensional array
 * B        : pointer to the host 2-dimensional array
 * nTile    : Number of memory tiles
 * inPlace  : if true the result is written in A otherwise a new 2-dimensional array is returned
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the pointer to the host 2-dimensional array containing the subtraction.
*/
__host__ ndarray* cudaMMSub(ndarray* A, ndarray* B, int nTile, bool inPlace, bool verbose);


/**
 * cudaMMProduct kernel performs the matrix product between:
 * 
 * - two matrices A (m x n) and B (n x p) 
 * - a matrix A (m x n) and a column vector B (n, 1)
 * - a row vector A (1, n) and a column vector B (n, 1)
 * - a column vector A (n, 1) and a row vector B (1, n)
 * 
 * Let A and B two n-dimensional array with the form described above, C = A * B
 * Due to the complexity of matrix product the cuda stream version of it is not provided, therefore 
 * this function doesn't support grid-level parallelism on data.
 * 
 * Parameters
 * ----------
 * A    : 2-dimensional array of the device
 * B    : 2-dimensional array of the device
 * C    : 2-dimensional array of the device in which the result is written
*/
__global__ void cudaMMProduct (ndarray A, ndarray B, ndarray C);

/**
 * cudaMMProduct host function is responsible for properly transfering memory and launching cudaMMProduct kernel.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 2-dimensional array
 * B        : pointer to the host 2-dimensional array
 * verbose  : if true the function prints out useful information about memory transfer and execution time
 * 
 * Returns
 * -------
 * The function returns the pointer to the host n-dimensional array containing the product in the following forms:
 * 
 * - a matrix (m x p) if inputs are in the form: A (m x n) and B (n x p)
 * - a column vector (m, 1) if inputs are in the form: A (m x n) and B (m, 1)
 * - a scalar value if inputs are in the form: A (1, n) and B (n, 1)
 * - matrix (n x n) if inputs are in the form: A (n, 1) and B (1, n)
*/
__host__ ndarray* cudaMMProduct(ndarray* A, ndarray* B, bool verbose);


/**
 * cudaEigenvectors function computes the first k largest cudaEigenvectors of a matrix.
 * To find an eigenvector the function ueses the power method.
 * 
 * Parameters
 * ----------
 * M        : pointer to the host 2-dimensional array
 * k        : Number of eingenvectors to be computed
 * tol      : convergence tolerance
 * MAXITER  : Number of maximum iterations
 * 
 * Returns
 * -------
 * The function returns the pointer to the host 2-dimensional array containing the k largest eigenvectors.
 * A column matrix is returned.
*/
__host__ ndarray* cudaEigenvectors(ndarray* M, int k, float tol, int MAXITER);


#endif