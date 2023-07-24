#include <cuda_runtime.h>
#include <stdio.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"
#include "cudalib/cudalinalg.cuh"

float HOST_TOT_TIME = 0;
float DEVICE_TOT_TIME = 0;

bool NDARRAY_CHECK(ndarray* A, ndarray* B)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: NDARRAY_CHECK, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
        if(A->data[i] != B->data[i])
            return false;

    return true;
}

bool SCALAR_CHECK(float a, float b)
{
    return a == b;
}

void random_init(ndarray* A) {
  
  for (int i = 0; i < A->shape[0] * A->shape[1]; i++) 
  {
    A->data[i] = (float)rand() / RAND_MAX;
  }
}

void VSD_TEST(FILE* fp, int nTest, int exp, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        //int n = (int) sqrt(m);

        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        ndarray* A = cuda_ndarrayHost(1 ,m);
        //ndarray* B = cuda_ndarrayHost(n ,n);
        
        random_init(A);
        //random_init(B);
        
        ndarray* C_h = vectorScalarDivision(A, 10.f, false);
        ndarray* C_d = cudaVSDivision(A, 10.f, 1, false, false);

        bool passed = NDARRAY_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        free_(C_h);
        cudaFreeHost_(C_d);

        if(verbose)
            printf("TEST: vectorScalarDivision vs cudaVSDivision,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f, %.4f\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, passed);
        
        fprintf(fp, "TEST: vectorScalarDivision vs cudaVSDivision,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f, %.4f\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, passed);
        fflush(fp);

    }
}

void ED_TEST(FILE* fp, int nTest, int exp, bool verbose)
{
    printf("\n\n");
    fprintf(fp, "\n\n");
    fflush(fp);

    for(int i = 0; i < nTest; i++)
    {
        int m = 1 << (exp + i);
        //int n = (int) sqrt(m);

        HOST_TOT_TIME = 0;
        DEVICE_TOT_TIME = 0;

        //printf("OK\n");
        ndarray* A = cuda_ndarrayHost(1 ,m);
        ndarray* B = cuda_ndarrayHost(1 ,m);
        //printf("OK\n");
        
        //random_init(A);
        //random_init(B);
        init(A, 2);
        init(B, 1);
        //printf("OK\n");
        
        float C_h = euclideanDistance(A, B);
        //printf("%f\n", C_h);
        float C_d = cudaEDistance(A, B, 1, false);
        //printf("%f\n", C_d);

        bool passed = SCALAR_CHECK(C_h, C_d);

        cudaFreeHost_(A);
        cudaFreeHost_(B);

        if(verbose)
            printf("TEST: euclideanDistance vs cudaEDistance,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f, %.4f\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, passed);
        
        fprintf(fp, "TEST: euclideanDistance vs cudaEDistance,\tNumber of elements: %d,\tsize (MB): %.2f,\ttime (s): %.4f, %.4f\tpassed: %d\n", m,  m * sizeof(float) / pow(2, 20), HOST_TOT_TIME, DEVICE_TOT_TIME / 1000, passed);
        fflush(fp);

    }
}


int main()
{   
    FILE* fp = fopen("test_results.txt", "w");

    //VSD_TEST(fp, 4, 25, true);
    ED_TEST(fp, 4, 25, true);

    fflush(fp);
    fclose(fp);




    return 0;
}