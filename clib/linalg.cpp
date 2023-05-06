#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linalg.h"

ndarray* matProduct(ndarray* A, ndarray* B)
{

    if(A->shape[1] != B->shape[0])
    {
        printf("ERROR: %d != %d; A columns must be equals to B rows\n", A->shape[1], B->shape[0]);
        return NULL;
    }

    double* mp_data = (double*)calloc(A->shape[0] * B->shape[1], sizeof(double));

    double* A_data = A->data;
    double* B_data = B->data;

    for(int i = 0; i < A->shape[0]; i++)
    {
        for(int j = 0; j < B->shape[1]; j++)
        {
            for (int n = 0; n < A->shape[1]; n++)
            {
                mp_data[B->shape[1] * i + j] += A_data[A->shape[1] * i + n] * B_data[B->shape[1] * n + j];
            }
        }
    }

    return new_ndarray(A->shape[0], B->shape[1], mp_data);
}

ndarray* vectorsProduct(ndarray* RV, ndarray* CV)
{
    if(RV->shape[1] != CV->shape[0])
    {
        printf("ERROR: %d != %d;\n", RV->shape[1], CV->shape[0]);
        return NULL;
    }

    double* vp_data = (double*)calloc(RV->shape[1] * CV->shape[0], sizeof(double));

    for(int i = 0; i < RV->shape[1]; i++)
    {
        for(int j = 0; j < CV->shape[0]; j++)
        {
            vp_data[RV->shape[1] * i + j] = RV->data[i] * CV->data[j];
        }
    }

    return new_ndarray(CV->shape[0], RV->shape[1], vp_data);
}

ndarray* matScalarProduct(ndarray* A, double scalar)
{
    double* msp_data = (double*)calloc(A->shape[0] * A->shape[1], sizeof(double));

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        msp_data[i] = A->data[i] * scalar;
    }

    return new_ndarray(A->shape[0], A->shape[1], msp_data);
}

ndarray* matSub(ndarray* A, ndarray* B)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR: %d != %d or %d != %d;\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        return NULL;
    }

    double* ms_data = (double*)calloc(A->shape[0] * A->shape[1], sizeof(double));

    for(int i = 0; i < A->shape[1] *  A->shape[0]; i++)
    {
       ms_data[i] = A->data[i] - B->data[i];
    }

    return new_ndarray(A->shape[0],  A->shape[1], ms_data);
}

ndarray* transpose(ndarray* A)
{

    double* AT_data = (double*)calloc(A->shape[0] * A->shape[1], sizeof(double));

    for (int i = 0; i < A->shape[0]; i++)
    {
        for (int j = 0; j< A->shape[1]; j++)
        {
            AT_data[A->shape[0] * j + i] = A->data[A->shape[1] * i + j];
        }
    }
    
    return new_ndarray(A->shape[1], A->shape[0], AT_data);
}

double norm(ndarray* X)
{
    double norm = 0;

    for(int i = 0; i < X->shape[0] * X->shape[1]; i++)
    {
        norm += pow(X->data[i], 2);
    }

    return sqrt(norm);
}

void normalize(ndarray* X, double nrm)
{   
    for(int i = 0; i < X->shape[0] * X->shape[1]; i++)
    {
        X->data[i] = X->data[i] / nrm; 
    }
}

float error(ndarray* x, ndarray* y)
{
    float e = 0;

    for(int i = 0; i < x->shape[0]; i++)
    {   
        e += pow(x->data[i] - y->data[i], 2);
    }

    return sqrt(e);
}

ndarray* eigenvectors(ndarray* M, int k, float tol)
{
    if(k > M->shape[1])
    {
        printf("ERROR: %d > %d; The number of eigenvectors must be at most the order of the matrix\n", k, M->shape[1]);
        return NULL;
    }

    ndarray* E = new_ndarray(M->shape[0], k, NULL);
    ndarray* eigenvector = new_ndarray(M->shape[1], 1, NULL);
    ndarray* Mt = copy(M);
    int MAX_ITER = 50;

    for(int t = 0; t < k; t++)
    {
        
        for(int i = 0; i < eigenvector->shape[0]; i++)
        {
            eigenvector->data[i] = 1;
        }

        float e = 0;
        int iter = 0;

        do{

            ndarray* eigenvectorNew = matProduct(Mt, eigenvector);
            normalize(eigenvectorNew, norm(eigenvectorNew));

            e = error(eigenvectorNew, eigenvector);
            free_(eigenvector);

            eigenvector = eigenvectorNew;

        }while(e > tol && iter++ < MAX_ITER);

        for(int i = 0; i < eigenvector->shape[0]; i++)
        {
            E->data[eigenvector->shape[0] * t + i] = eigenvector->data[i];
        }

        ndarray* me = matProduct(Mt, eigenvector);
        ndarray* meT = transpose(me);
        free_(me);

        ndarray* eigenvalue = matProduct(meT, eigenvector);
        free_(meT);


        ndarray* eigenvectorT = transpose(eigenvector);
        ndarray* vp = vectorsProduct(eigenvectorT, eigenvector);
        free_(eigenvectorT);

        ndarray* msp = matScalarProduct(vp, eigenvalue->data[0]);
        free_(vp);
        free_(eigenvalue);

        ndarray* Mtmp = matSub(Mt, msp);
        free_(Mt);
        free_(msp);

        Mt = Mtmp;
    }

    free_(Mt);
    free_(eigenvector);

    return E;
}