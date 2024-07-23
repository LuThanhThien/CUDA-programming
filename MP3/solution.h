// wb.h

#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>
#include <corecrt_math.h>
#include <crt/host_config.h>

#define EPSILON 1e-2


float *wbImport(const char *file, int *rows, int* cols) {
    FILE *fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "Unable to open file %s\n", file);
        exit(EXIT_FAILURE);
    }
    
    // Read the rows from the first line of the file
    fscanf(fp, "%d %d", rows, cols);
    printf("rows of array from %s is %d\n", file, *rows);
    printf("cols of array from %s is %d\n", file, *cols);

    // Allocate memory for the data
    float *data = (float *)malloc((*rows) * (*cols) * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Read the floating-point numbers
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            fscanf(fp, "%f", &data[i * (*cols) + j]);
        }
    }
    fclose(fp);
    printf("Successfully read data from file %s\n", file);
    return data;
}

void wbSolution(wbArg_t args, float *output, int rows, int cols) {
    int resultRows;
    int resultCols;
    float* results = (float *)wbImport(wbArg_getOutputFile(args), &resultRows, &resultCols);
    if (rows != resultRows || cols != resultCols) {
        fprintf(stderr, "Result height or width not matched");
        exit(EXIT_FAILURE);
    }
   
    for (int i = 0; i < rows; i++) {
        if (fabs(output[i] - results[i]) > EPSILON) {
            fprintf(stderr, "Result not matched at element number %d: your output = %f :: actual output = %f\n", i, output[i], results[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("All matched!\n");
}


#endif // SOLUTION_H
