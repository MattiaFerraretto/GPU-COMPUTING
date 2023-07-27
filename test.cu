#include <cuda_runtime.h>
#include <stdio.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"
#include "cudalib/cudalinalg.cuh"

float HOST_TOT_TIME = 0;
float DEVICE_TOT_TIME = 0;

/**
 * NDARRAY_CHECK function checks if two n-dimensional array are equal with a tollerance error of 1e-2
*/
bool NDARRAY_CHECK(ndarray* A, ndarray* B)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: NDARRAY_CHECK, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
        if(abs(A->data[i] - B->data[i]) > 1e-2)
            return false;
        

    return true;
}

/**
 * SCALAR_CHECK function checks if two scalar are equals
*/
bool SCALAR_CHECK(float a, float b)
{
    return a == b;
}

/**
 * random_init function initializes randomly a n-dimensional array
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
 * Vector scalar division test
*/
void VSD_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);

        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(1 ,m);
        
        random_init(A);
        
        ndarray* C_h = vectorScalarDivision(A, 10.f, false);
        ndarray* C_d = cudaVSDivision(A, 10.f, nTile, false, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: vectorScalarDivision vs cudaVSDivision,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: vectorScalarDivision vs cudaVSDivision,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Eclidean distance test
*/
void ED_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);

        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(1 ,m);
        ndarray* B = cuda_ndarrayHost(1 ,m);
        
        random_init(A);
        random_init(B);
        
        float C_h = euclideanDistance(A, B);
        float C_d = cudaEDistance(A, B, nTile, false);

        bool passed = SCALAR_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        cudaFreeHost_(B);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: euclideanDistance vs cudaEDistance,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: euclideanDistance vs cudaEDistance,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Matrix scalr product test 
*/
void MSP_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        int n = (int) sqrt(m);


        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(n ,n);
        
        random_init(A);
        
        ndarray* C_h = matScalarProduct(A, 10.f, false);
        ndarray* C_d = cudaMSProduct(A, 10.f, nTile, false, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: matScalarProduct vs cudaMSProduct,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: matScalarProduct vs cudaMSProduct,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Matrix transpose test
*/
void MT_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        int n = (int) sqrt(m);


        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(n ,n);
        
        random_init(A);
        
        ndarray* C_h = matTranspose(A);
        ndarray* C_d = cudaMTranspose(A, nTile, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: matTranspose vs cudaMTranspose,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: matTranspose vs cudaMTranspose,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Matrix subtraction test
*/
void MS_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        int n = (int) sqrt(m);


        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(n ,n);
        ndarray* B = cuda_ndarrayHost(n ,n);
        
        random_init(A);
        random_init(B);
        
        ndarray* C_h = matSub(A, B, false);
        ndarray* C_d = cudaMMSub(A, B, nTile, false, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        cudaFreeHost_(B);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: matSub vs cudaMMSub,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: matSub vs cudaMMSub,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Matrix product test
*/
void MP_TEST(FILE* fp, int nTest, int exp, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        int n = (int) sqrt(m);

        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(n ,n);
        ndarray* B = cuda_ndarrayHost(n ,n);
        
        random_init(A);
        random_init(B);
        
        ndarray* C_h = matProduct(A, B);
        ndarray* C_d = cudaMMProduct(A, B, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        cudaFreeHost_(B);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: matProduct vs cudaMMProduct,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: matProduct vs cudaMMProduct,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

/**
 * Eigenvectors test
*/
void E_TEST(FILE* fp, int nTest, int exp, int nTile, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        int n = (int) sqrt(m);


        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A_h = cuda_ndarrayHost(n ,n);
        
        random_init(A_h);
        
        ndarray* C_h = eigenvectors(A_h, 20, 1e-6, 50);
        ndarray* C_d = cudaEigenvectors(A_h, 20, 1e-6, 50);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A_h);
        free_(C_h);
        cudaFreeHost_(C_d);

        float speedup = HOST_TOT_TIME / (DEVICE_TOT_TIME / 1000);

        if(verbose)
            printf("TEST: eigenvectors vs cudaEigenvectors,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        
        fprintf(fp, "TEST: eigenvectors vs cudaEigenvectors,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f vs %.4f,\tGPU speedup: %.4f,\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, speedup, passed);
        fflush(fp);

    }
}

int main()
{   
    FILE* fp = fopen("test_results.txt", "w");

    int nTest = 4;
    int exp = 24;
    int nTile = 1;
    bool verbose = true;

    VSD_TEST(fp, nTest, exp, nTile, verbose);
    ED_TEST(fp, nTest, exp, nTile, verbose);
    MSP_TEST(fp, nTest, exp, nTile, verbose);
    MT_TEST(fp, nTest, exp, nTile, verbose);
    MS_TEST(fp, nTest, exp, nTile, verbose);
    MP_TEST(fp, nTest, exp, verbose);
    E_TEST(fp, nTest, exp, nTile, verbose);

    fflush(fp);
    fclose(fp);


    return 0;
}