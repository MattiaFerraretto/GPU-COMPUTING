#ifndef linalg_h
#define linalg_h

#include "ndarray.h"


/**
 * matProduct method computes matrix multiplication between:
 * 
 * - two matrices A (m x n) and B (n x p);
 * - a matrix A (m x n) and a column vector (n x 1);
 * - a row vector (1 x n) and a column vector (n x 1);
 * 
 * Parameters
 * -----------
 * 
 * A :  pointer to the first n dimensional array of doubles
 * B :  pointer to the second n dimensional array of doubles
 * 
 * Returns
 * -------
 * 
 * This function returns a pointer to the n dimensional array containing the matrix product.
 * When the product is between:
 * 
 * - two matrices: it returns an array containing m x p elments;
 * - a matrix and a columns vector: it retruns a row array (1 x n)
 * - a row vector and a column vector: it returns a single scalar value
 * 
*/

ndarray* matProduct(ndarray* A, ndarray* B);

ndarray* vectorsProduct(ndarray* RV, ndarray* CV);

ndarray* matScalarProduct(ndarray* A, float scalar);

ndarray* matSub(ndarray* A, ndarray* B);

ndarray* transpose(ndarray* A);

double norm(ndarray* X);

void normalize(ndarray* X, float nrm);

float error(ndarray* x, ndarray* y);

ndarray* eigenvectors(ndarray* M, int k, float tol);

#endif