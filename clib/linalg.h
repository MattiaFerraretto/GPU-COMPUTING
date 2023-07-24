#ifndef linalg_h
#define linalg_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <time.h>

#include "ndarray.h"



/**
 * vectorScalarDivision function computes the element-wise division between a vector and a scalar value.
 * Let A be a vector,  
 * Parameters
 * ----------
 * A        : 1-dimensional array
 * value    : scalar value
 * inPlace  : if true the result is written in A otherwise a new 1-dimensional array is returned
 * 
 * Returns
 * -------
 * The function returns the pointer to the 1-dimension array cointaing the element-wise division
 * in the case inPlace option is false, otherwise NULL
*/
ndarray* vectorScalarDivision(ndarray* A, float value, bool inPlace);

/**
 * euclideanDistance function computes the eclidean distance between two vectors.
 * In the case one of them is the origin vector, Frobenius norm is returned.
 * 
 * Parameters
 * ----------
 * A    : 1-dimensional array
 * B    : 1-dimensional array
 * 
 * Returns
 * -------
 * The function returns the eucledian distance between two vectors 
*/
float euclideanDistance(ndarray* A, ndarray* B);

/**
 * matScalarProduct function computes the element-wise product between a vector and a scalar value.
 * 
 * Parameters
 * ----------
 * A        : 1-dimensional array
 * scalar   : scalar value
 * inPlace  : if true the result is written in A otherwise a new 1-dimensional array is returned
 * 
 * Retruns
 * -------
 * The function returns the pointer to the 1-dimensional array containing the element-wise product
 * in the case inPlace option is false, otherwise NULL
*/
ndarray* matScalarProduct(ndarray* A, float scalar, bool inPlace);

/**
 * matTranspose function compues the transpose of a matrix.
 * 
 * Parameters
 * ----------
 * A    : 2-dimensional array
 * 
 * Retruns
 * -------
 * The function returns the pointer to the 2-dimensional array cointaing the trasnspose matrix.
*/
ndarray* matTranspose(ndarray* A);

/**
 * matSub function computes the element-wise subtraction between two matrices.
 * 
 * Parameters
 * ----------
 * A        : 2-dimensional array
 * B        : 2 dimensional array
 * inPlace  : if true the result is written in A otherwise a new 1-dimensional array is returned
 * 
 * Returns
 * -------
 * The function returns the pointer to the 2-dimensional array containing the element-wise subtraction
 * in the case inPlace option is false, otherwise NULL
*/
ndarray* matSub(ndarray* A, ndarray* B, bool inPlace);

/**
 * matProduct function computes matrix multiplication between:
 * 
 * - two matrices A (m x n) and B (n x p)
 * - a matrix A (m x n) and a column vector B (n x 1)
 * - a row vector A (1 x n) and a column vector B (n x 1)
 * - a column vector A (n x 1) and a row vector B (1 x n)
 * 
 * Parameters
 * -----------
 * A :  pointer to the first 2-dimensional array
 * B :  pointer to the second 2-dimensional array
 * 
 * Returns
 * -------
 * The function returns the pointer to the n-dimensional array containing the product in the following forms:
 * 
 * - a matrix (m x p) if inputs are in the form: A (m x n) and B (n x p)
 * - a column vector (m, 1) if inputs are in the form: A (m x n) and B (m, 1)
 * - a scalar value if inputs are in the form: A (1, n) and B (n, 1)
 * - matrix (n x n) if inputs are in the form: A (n, 1) and B (1, n)
*/
ndarray* matProduct(ndarray* A, ndarray* B);


/**
 * eigenvectors function computes the first k largest eigenvectors of a matrix.
 * To find an eigenvector the function ueses the power method.
 * 
 * Parameters
 * ----------
 * A        : pointer to the host 2-dimensional array
 * k        : Number of eingenvectors to be computed
 * tol      : convergence tolerance
 * MAXITER  : Number of maximum iterations
 * 
 * Returns
 * -------
 * The function returns the pointer to the 2-dimensional array containing the k largest eigenvectors.
 * A column matrix is returned.
*/
ndarray* eigenvectors(ndarray* A, int k, float tol, int MAXITER);

#endif