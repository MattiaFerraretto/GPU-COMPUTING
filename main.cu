#include <stdio.h>
#include <stdlib.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"
#include "cudalib/cudalinalg.cuh"

float HOST_TOT_TIME = 0;
float DEVICE_TOT_TIME = 0;

/**
 * Host PCA version
*/
ndarray* PCA(ndarray* M, int k)
{
    ndarray* MT = matTranspose(M);

    ndarray* cov = matProduct(MT, M);
    free_(MT);

    ndarray* E = eigenvectors(cov, k, 1e-10, 1000);

    ndarray* mpca = matProduct(M, E);
    free_(E);

    return mpca;
}

/**
 * Cuda PCA version
*/
ndarray* cudaPCA(ndarray* M, int k)
{
    ndarray* MT = cudaMTranspose(M, 1, false);

    ndarray* cov = cudaMMProduct(MT, M, false);
    cudaFreeHost_(MT);

    ndarray* E = cudaEigenvectors(cov, k, 1e-10, 1000);

    ndarray* mpca = cudaMMProduct(M, E, false);
    cudaFreeHost_(E);

    return mpca;
}

void from_file_example(){

    ndarray* M = new_ndarray(569, 30);

    csv2ndarry(M, "./DATA/breast_cancer.csv", ",");

    printShape(M);

    ndarray* Mpca = PCA(M, 15);
    printShape(Mpca);

    ndarray2csv("out.csv", Mpca, ",");
}

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

/**
 * PCA test
*/
float PCA_TEST(ndarray* A, ndarray* B)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: PCA_TEST, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    int n = A->shape[0] * A->shape[1];
    double err = 0.f;

    for(int i = 0; i < n; i++)
    {
        err += (A->data[i] - B->data[i]) * (A->data[i] - B->data[i]);
    }

    return err == 0 ? 0 : sqrt(err / n);
}

int main(){

    int m = 1 << 20;

    int n = (int) sqrt(m);

    ndarray* M = cuda_ndarrayHost(n, n);
   
    random_init(M);

    ndarray* pca = PCA(M, 20);
    ndarray* cuda_pca = cudaPCA(M, 20);

    float err = PCA_TEST(pca, cuda_pca);
    
    cudaFreeHost_(M);
    free_(pca);
    cudaFreeHost_(cuda_pca);

    float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

    printf("TEST: PCA vs cudaPCA,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f, \tGPU speedup: %.4f,\tExpected error: %f\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, err);
    

    return 0;
}