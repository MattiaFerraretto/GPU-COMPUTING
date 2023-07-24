#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"
#include "cudalib/cudalinalg.cuh"

float HOST_TOT_TIME = 0;
float DEVICE_TOT_TIME = 0;

ndarray* PCA(ndarray* M, int k)
{
    ndarray* MT = matTranspose(M);

    ndarray* cov = matProduct(MT, M);
    free_(MT);

    ndarray* E = eigenvectors(cov, k, 1e-6, 50);

    ndarray* mpca = matProduct(M, E);
    free_(E);

    return mpca;
}

ndarray* cudaPCA(ndarray* M, int k)
{
    ndarray* MT = cudaMTranspose(M, 1, false);

    ndarray* cov = cudaMMProduct(MT, M, false);
    cudaFreeHost_(MT);

    ndarray* E = cudaEigenvectors(cov, k, 1e-6, 50);

    ndarray* mpca = cudaMMProduct(M, E, false);
    cudaFreeHost_(E);

    return mpca;
}

void test1(){

    ndarray* M = new_ndarray(4, 2);

    M->data[0] = 1;
    M->data[1] = 2; 
    M->data[2] = 2; 
    M->data[3] = 1; 
    M->data[4] = 3; 
    M->data[5] = 4; 
    M->data[6] = 4; 
    M->data[7] = 3; 

    //ndarray* Mpca = PCA(M, 1);

    //print(Mpca);
}

void test2(){

    //ndarray* data = csv2ndarry("./DATA/breast_cancer.csv", 569, 30, ",");
    //printShape(data);

    ndarray* data = new_ndarray(569, 30);

    csv2ndarry(data, "./DATA/breast_cancer.csv", ",");

    printShape(data);

   // ndarray* Mpca = PCA(data, 5);
    //printShape(Mpca);

    //ndarray2csv("out.csv", Mpca, ",");

    ndarray2csv("out.csv", data, ",");
}

void initTest(ndarray* test){
    test->data[0] = 3;
    test->data[1] = 2;
    test->data[2] = 2;
    test->data[3] = 6;
}

void random_init(ndarray* A) {
  
  for (int i = 0; i < A->shape[0] * A->shape[1]; i++) 
  {
    A->data[i] = (float)rand() / RAND_MAX;
  }

  printf("ARRAY INIT!!\n");
}

int main(){

    int m = 1 << 25;

    printf("m: %d, MB: %f\n", m,  m * sizeof(float) / pow(2, 20));

    int n = (int) sqrt(m);
    printf("%d\n", n);

    ndarray* M = cuda_ndarrayHost(n, n);
    //csv2ndarry(M, "./DATA/breast_cancer.csv", ",");
    //initTest(M);
    random_init(M);
    //print(M);

    ndarray* pca = PCA(M, 15);
    //print(pca);
    printf("Host total time: %.10f\n", HOST_TOT_TIME);

    printf("--------------- CUDA RESULTS ---------------\n");

    ndarray* cuda_pca = cudaPCA(M, 15);
    //print(cuda_pca);

    printf("Device total time: %.10f\n", DEVICE_TOT_TIME / 1000);
    

    return 0;
}