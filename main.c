#include <stdio.h>
#include <stdlib.h>

#include "clib/ndarray.h"
#include "clib/linalg.h"


ndarray* PCA(ndarray* M, int d)
{
    ndarray* MT = transpose(M);

    ndarray* cov = matProduct(MT, M);
    free_(MT);

    ndarray* E = eigenvectors(cov, d, 1e-6);

    if(E == NULL)
        return NULL;

    ndarray* mpca = matProduct(M, E);
    free_(E);

    return mpca;
}

void test1(){

    double M_data[8] = {1, 2, 2, 1, 3, 4, 4, 3};
    ndarray* M = new_ndarray(4, 2, M_data);

    ndarray* Mpca = PCA(M, 1);

    print(Mpca);
}

void test2(){
    ndarray* data = csv2ndarry("./DATA/breast_cancer.csv", 569, 30, ",");
    printShape(data);

    ndarray* Mpca = PCA(data, 5);
    printShape(Mpca);

    ndarray2csv(Mpca, "out.csv", ",");
}

int main(){

    //test1();
    test2();

    return 0;
}