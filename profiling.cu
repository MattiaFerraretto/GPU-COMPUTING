#include <stdio.h>
#include <stdlib.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"
#include "cudalib/cudalinalg.cuh"

float HOST_TOT_TIME = 0;
float DEVICE_TOT_TIME = 0;

/**
 * Random initializer for tests
*/
void random_init(ndarray* A)
{
    srand(time(NULL));

    for (int i = 0; i < A->shape[0] * A->shape[1]; i++) 
    {
        A->data[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{

    int m = 1 << 28;

    int n = (int) sqrt(m);

    printf("Number of elements: %d,\tsize (MB): %.2f\n", m,  m * sizeof(float) / pow(2, 20));

    ndarray* C;

    ndarray* A = cuda_ndarrayHost(1, m);
    ndarray* B = cuda_ndarrayHost(1, m);

    random_init(A);
    random_init(B);

    C = cudaVSDivision(A, 10.f, 1, false, false);
    cudaFreeHost_(C);

    cudaEDistance(A, B, 1, false);
    
    cudaFreeHost_(A);
    cudaFreeHost_(B);

    A = cuda_ndarrayHost(n, n);
    B = cuda_ndarrayHost(n, n);

    random_init(A);
    random_init(B);

    C = cudaMSProduct(A, 10.f, 1, false, false);
    cudaFreeHost_(C);

    C = cudaMTranspose(A, 1, false);
    cudaFreeHost_(C);

    C = cudaMMSub(A, B, 1, false, false);
    cudaFreeHost_(C);

    C = cudaMMProduct(A, B, false);
    cudaFreeHost_(C);


    return 0;
}