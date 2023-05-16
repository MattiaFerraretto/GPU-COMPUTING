#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ndarray.h"


ndarray* new_ndarray(int rows, int columns, float* data)
{
    ndarray* a = (ndarray*)malloc(sizeof(ndarray));
    int* shape = (int*)calloc(2, sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    a->shape = shape;

    if(data == NULL)
        a->data = (float*)calloc(rows * columns, sizeof(float));
    else
        a->data = data;
    
    return a;
}


void free_(ndarray* A)
{
    free(A->shape);
    free(A->data);
    free(A);
}

ndarray* copy(ndarray* A)
{
    float* data = (float*)calloc(A->shape[0] * A->shape[1], sizeof(float));

    for(int i = 0; i < A->shape[0] * A->shape[1]; i++)
    {
        data[i] = A->data[i];
    }

    return new_ndarray(A->shape[0], A->shape[1], data);
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
        printf("ERROR; print: ndarray pointer null\n");
        return ;
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
        printf("ERROR; print: ndarray pointer null\n");
        return ;
    }

    printf("%p's shape: (%d, %d)\n\n", A, A->shape[0], A->shape[1]);
}

ndarray* csv2ndarry(char* filePath, int rows, int features, char* delimiter)
{
    FILE* fp = fopen(filePath, "r");
    float* data_points = (float*)calloc(rows * features, sizeof(float));
    char* buffer = (char*)malloc(1024 * sizeof(char));

    if(fp == NULL)
    {
        printf("ERROR: file not found.\n");
        return NULL;
    }

    for(int i = 0; i < rows; i++)
    {
        fgets(buffer, 1024 * sizeof(char), fp);
        char* token = strtok(buffer, delimiter);

        for(int j = 0; j < features; j++)
        {
            sscanf(token, "%f", data_points + (features * i + j));
            token = strtok(NULL, delimiter);
        }
    }

    free(buffer);
    fclose(fp);

    return new_ndarray(rows, features, data_points);
}

void ndarray2csv(ndarray* A, char* filePath, char* delimiter)
{
    if(A == NULL)
    {
        printf("ERROR; print: ndarray pointer null\n");
        return ;
    }

    FILE* fp = fopen(filePath, "w");

    for(int i = 0; i < A->shape[0]; i++)
    {
        for(int j = 0; j < A->shape[1]; j++)
        {
            fprintf(fp, "%f", A->data[A->shape[1] * i + j]);
            fflush(fp);
            
            if(j != A->shape[1] - 1)
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



