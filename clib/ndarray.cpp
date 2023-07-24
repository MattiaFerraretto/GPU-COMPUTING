
#include "ndarray.h"


ndarray* new_ndarray(int rows, int columns)
{
    ndarray* a = (ndarray*)malloc(sizeof(ndarray));
    int* shape = (int*)calloc(2, sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    a->shape = shape;
    a->data = (float*)calloc(rows * columns, sizeof(float));

    return a;
}

void free_(ndarray* A)
{
    free(A->shape);
    free(A->data);
    free(A);
}

void init(ndarray* A, float value)
{
    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        A->data[i] = value;
    }
}

void reshape(ndarray* A, int rows, int columns)
{
    A->shape[0] = rows;
    A->shape[1] = columns;
}

void print(ndarray* A)
{   
    if(A == NULL)
    {
        printf("ERROR: ndarray pointer null\n");
        exit(EXIT_FAILURE);
    }

    printf("Pointer: %p\nShape: (%d, %d)\n\n", A, A->shape[0], A->shape[1]);
    printf("[");

    for( int i =0; i < A->shape[0]; i++)
    {   
        if(i > 0)
            printf(" ");

        for( int j = 0; j < A->shape[1]; j++)
        {   
            printf("%-*.*f", 10, 6, A->data[A->shape[1] * i + j]);

            //if(j != A->shape[1] - 1)
            //    printf("\t");
        }

        if(i != A->shape[0] - 1)
            printf("\n");
    }

    printf("]\n\n");

}

void printShape(ndarray* A)
{
    if(A == NULL)
    {
        printf("ERROR: ndarray pointer null\n");
        exit(EXIT_FAILURE);
    }

    printf("%p's shape: (%d, %d)\n\n", A, A->shape[0], A->shape[1]);
}

void csv2ndarry(ndarray* dest, const char* src, const char* delimiter)
{
    FILE* fp = fopen(src, "r");
    char* buffer = (char*)malloc(1024 * sizeof(char));

    if(fp == NULL)
    {
        printf("ERROR: file not found.\n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < dest->shape[0]; i++)
    {
        fgets(buffer, 1024 * sizeof(char), fp);
        char* token = strtok(buffer, delimiter);

        for(int j = 0; j < dest->shape[1]; j++)
        {
            sscanf(token, "%f", &dest->data[dest->shape[1] * i + j]);
            token = strtok(NULL, delimiter);
        }
    }

    free(buffer);
    fclose(fp);

}

void ndarray2csv(const char* dest, ndarray* src, const char* delimiter)
{
    if(src == NULL)
    {
        printf("ERROR: ndarray pointer null\n");
        exit(EXIT_FAILURE);
    }

    FILE* fp = fopen(dest, "w");

    for(int i = 0; i < src->shape[0]; i++)
    {
        for(int j = 0; j < src->shape[1]; j++)
        {
            fprintf(fp, "%f", src->data[src->shape[1] * i + j]);
            fflush(fp);
            
            if(j != src->shape[1] - 1)
            {
                fprintf(fp, "%s", delimiter);
                fflush(fp);
            }
        }

        fprintf(fp, "\n");
        fflush(fp);
    }

    fclose(fp);

}



