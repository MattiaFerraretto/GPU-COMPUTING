
#include "linalg.h"

extern float HOST_TOT_TIME;

/**
 * See the documentation of matProduct function in the file linalg.h 
*/
ndarray* matProduct(ndarray* A, ndarray* B)
{

    if(A->shape[1] != B->shape[0])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: matProduct, ", __FILE__, __LINE__);
        printf("reason: %d != %d ; Columns of A must be equals to rows of B.\n", A->shape[1], B->shape[0]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = new_ndarray(A->shape[0], B->shape[1]);
    clock_t start, end;

    start = clock();
    for(int i = 0; i < A->shape[0]; i++)
    {
        for(int j = 0; j < B->shape[1]; j++)
        {
            for (int n = 0; n < A->shape[1]; n++)
            {
                C->data[B->shape[1] * i + j] += A->data[A->shape[1] * i + n] * B->data[B->shape[1] * n + j];
            }
        }
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);

    return C;
}

/**
 * See the documentation of matScalarProduct function in the file linalg.h 
*/
ndarray* matScalarProduct(ndarray* A, float scalar, bool inPlace)
{
    ndarray* C =  inPlace? A : new_ndarray(A->shape[0], A->shape[1]);
    clock_t start, end;

    start = clock();
    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        C->data[i] = A->data[i] * scalar;
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);

    return inPlace? NULL : C;
}

/**
 * See the documentation of matSub function in the file linalg.h 
*/
ndarray* matSub(ndarray* A, ndarray* B, bool inPlace)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: matSub, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = inPlace? A : new_ndarray(A->shape[0], A->shape[1]);
    clock_t start, end;

    start = clock();
    for(int i = 0; i < A->shape[1] *  A->shape[0]; i++)
    {
       C->data[i] = A->data[i] - B->data[i];
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);

    return inPlace ? NULL : C;
}

/**
 * See the documentation of matTranspose function in the file linalg.h 
*/
ndarray* matTranspose(ndarray* A)
{
    ndarray* AT = new_ndarray(A->shape[1], A->shape[0]);
    clock_t start, end;

    start = clock();
    for (int i = 0; i < A->shape[0]; i++)
    {
        for (int j = 0; j< A->shape[1]; j++)
        {
            AT->data[A->shape[0] * j + i] = A->data[A->shape[1] * i + j];
        }
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);
    
    return AT;
}

/**
 * See the documentation of vectorScalarDivision function in the file linalg.h 
*/
ndarray* vectorScalarDivision(ndarray* A, float value, bool inPlace)
{
    if(A->shape[0] > 1 && A->shape[1] > 1)
    {
        printf("ERROR:: File: %s, Line: %d, Function name: cudaVSDivision, ", __FILE__, __LINE__);
        printf("reason: (%d, %d) != (1, n) || (%d, %d) != (n, 1); A must be a row vector (1, n) or a column vector (n, 1).\n", A->shape[0], A->shape[1], A->shape[0], A->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = inPlace ? A : new_ndarray(A->shape[0], A->shape[1]);
    clock_t start, end;

    start = clock();
    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        C->data[i] = A->data[i] / value; 
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);

    return inPlace ? NULL : C;
}

/**
 * See the documentation of euclideanDistance function in the file linalg.h 
*/
float euclideanDistance(ndarray* A, ndarray* B)
{
    // (n, 1) (1, n)
    // (1, n) (n, 1)
    // (1, n) (1, n)
    // (n, 1) (n, 1)
    bool check1 = A->shape[0] > 1 && !(B->shape[0] > 1) && A->shape[0] != B->shape[1];
    bool check2 = !(A->shape[0] > 1) && B->shape[0] > 1 && A->shape[1] != B->shape[0];
    bool check3 = !(A->shape[0] > 1) && !(B->shape[0] > 1) && A->shape[1] != B->shape[1];
    bool check4 = A->shape[0] > 1 && B->shape[0] > 1 && A->shape[0] != B->shape[0];

    if(check1 || check2 || check3 || check4)
    {
        printf("ERROR:: File: %s, Line: %d, Function name: euclideanDistance, ", __FILE__, __LINE__);
        printf("reason: (%d, %d) != (%d, %d); incompatible shape, A and B must be row vectors (1, n) or column vectors (n, 1).\n", A->shape[0], A->shape[1], B->shape[0], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    double distance = 0.f;
    clock_t start, end;

    start = clock();
    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {   
        distance += (A->data[i] - B->data[i]) * (A->data[i] - B->data[i]);
    }
    end = clock();

    HOST_TOT_TIME += ((float)(end - start) / CLOCKS_PER_SEC);

    return distance == 0 ? 0 : sqrt(distance);
}

/**
 * See the documentation of eigenvectors function in the file linalg.h 
*/
ndarray* eigenvectors(ndarray* A, int k, float tol, int MAX_ITER)
{
    if(k > A->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: eigenvectors, ", __FILE__, __LINE__);
        printf("reason: %d > %d; The numbers of eigenvectors (k) must be at most equals to the rank of the matrix.\n", k, A->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* E = new_ndarray(k, A->shape[1]);
    ndarray* O = new_ndarray(A->shape[1], 1);

    float sqrterr = 0.f;


    for(int i = 0; i < k; i++)
    {
        int iter = 0;
        
        ndarray* x = new_ndarray(A->shape[1], 1);
        init(x, 1);

        ndarray* eigvc;
        do{

            eigvc = matProduct(A, x);

            float norm = euclideanDistance(eigvc, O);
            vectorScalarDivision(eigvc, norm, true);

            sqrterr = euclideanDistance(eigvc, x);
            free_(x);

            x = eigvc;

        }while(sqrterr > tol && iter++ < MAX_ITER);

        memcpy((void*)&E->data[i * A->shape[1]], (void*)eigvc->data, eigvc->shape[0] * sizeof(float));

        ndarray* axeigvc = matProduct(A, eigvc);
        ndarray* eigvcT = matTranspose(eigvc);

        ndarray* eigva = matProduct(eigvcT, axeigvc);
        free_(axeigvc);

        ndarray* m = matProduct(eigvc, eigvcT);
        free_(eigvc);
        free_(eigvcT);

        matScalarProduct(m, eigva->data[0], true);
        free_(eigva);
        
        matSub(A, m, true);
        free_(m);
    }

    ndarray* ET = matTranspose(E);
    free_(E);
    free_(O);

    return ET;
}