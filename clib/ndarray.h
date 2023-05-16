#ifndef ndarray_h
#define ndarray_h

typedef struct ndarray_
{   
    int* shape;
    float* data;

} ndarray;

ndarray* new_ndarray(int rows, int columns, float* data);

void free_(ndarray* A);

ndarray* copy(ndarray* A);

void reshape(ndarray* A, int rows, int columns);

/**
 * print method prints to the standard output the n dimensional array passed as parameter, shaping it 
 * with respect to the number of rows and columns.
 * 
 * Parameters
 * ----------
 * 
 * A : pointer to the n dimnesional array to print
 * 
*/
void print(ndarray* A);

void printShape(ndarray* A);

ndarray* csv2ndarry(char* filePath, int rows, int features, char* delimiter);

void ndarray2csv(ndarray* A, char* filePath, char* delimiter);


#endif