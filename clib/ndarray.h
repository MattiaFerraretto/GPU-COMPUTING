#ifndef ndarray_h
#define ndarray_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/**
 * ndarray_ struct represents n-dimensional array where:
 * - the shape pointer refers to the sizes of n-dimensional array
 * - the data pointer refers to the elements of n-dimensional array
 * 
 * PAY ATTENTION : the strucut is designed to handle n-dimensional array, 
 *                 but libraries implemented handle only 1 and 2 dimensional
 *                 ones
*/
typedef struct ndarray_
{   
    int* shape;
    float* data;

} ndarray;

/**
 * new_ndarray function creates a new n-dimensional array (rows x columns)
 * 
 * Parameters
 * ----------
 * rows     : Number of rows
 * columns  : Number of columns
 * 
 * Returns
 * -------
 * The function returns the pointer to the n-dimensional array created 
*/
ndarray* new_ndarray(int rows, int columns);

/**
 * free_ function frees the space allocated for the n-dimensional array
 * 
 * Parameters
 * ----------
 * A    : pointer to the n-dimensional array to be deallocated
*/
void free_(ndarray* A);

/**
 * init function sets all entries in the n-dimensional array to the value passed as parameter
 * 
 * Parameters
 * ----------
 * A        : pointer to the n-dimensional array
 * value    : value to be set
*/
void init(ndarray* A , float value);

/**
 * reshape function reshapes the n-dimensional array with the new row and column values
 * 
 * Parameters
 * ----------
 * A        : pointer to the n-dimensional array
 * rows     : new rows value
 * columns  : new columns value
*/
void reshape(ndarray* A, int rows, int columns);

/**
 * print function prints to the standard output the n-dimensional array passed as parameter, shaping it 
 * with respect to the number of rows and columns.
 * 
 * Parameters
 * ----------
 * A    : pointer to the n-dimensional array to print
*/
void print(ndarray* A);

/**
 * printShape function prints the shape of the n-dimensional array
 * 
 * Parameters
 * ----------
 * A    : pointer to the n-dimensional array
*/
void printShape(ndarray* A);

/**
 * csv2ndarry reads from a cvs an n-dimensional array and loads it into the destination
 * 
 * Parameters
 * ----------
 * dest         : pointer to the target n-dimensional array 
 * src          : file path string
 * delimiter    : csv delimiter string
*/
void csv2ndarry(ndarray* dest, const char* src, const char* delimiter);

/**
 * ndarray2csv function writes the n-dimensional array pointed to a csv file
 * 
 * Parameters
 * ----------
 * dest         : file path string
 * src          : pointer to the n-dimensional array
 * delimiter    : csv delimiter string
*/
void ndarray2csv(const char* dest, ndarray* src, const char* delimiter);

#endif