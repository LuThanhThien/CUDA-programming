// wb.h

#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>
#include <crt/host_config.h>    

#define EPSILON 1e-6 

float *wbImport(const char *file, int *length) {
    FILE *fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "Unable to open file %s\n", file);
        exit(EXIT_FAILURE);
    }
    
    // Read the length from the first line of the file
    fscanf(fp, "%d", length);
    printf("Length of array from %s is %d\n", file, *length);

    // Allocate memory for the data
    float *data = (float *)malloc((*length) * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Read the floating-point numbers
    for (int i = 0; i < *length; i++) {
        fscanf(fp, "%f", &data[i]);
    }
    fclose(fp);
    printf("Successfully read data from file %s\n", file);
    return data;
}

void wbSolution(wbArg_t args, float *output, int length) {
    int resultLength;
    float* results = (float *)wbImport(wbArg_getOutputFile(args), &resultLength);
    if (length != resultLength) {
        fprintf(stderr, "Result length not matched");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < length; i++) {
        if (fabs(output[i] - results[i]) > EPSILON) {
            fprintf(stderr, "Result not matched at element number %d: your output = %f :: actual output = %f\n", i, output[i], results[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("All matched!\n");
}


#endif // SOLUTION_H
