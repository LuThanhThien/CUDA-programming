// wb.h

#ifndef WB_H
#define WB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <corecrt_math.h>

#define EPSILON 1e-6 
#define TRACE "TRACE"
#define GPU "GPU"

typedef struct {
    char **args;
    int count;
} wbArg_t;

wbArg_t wbArg_read(int argc, char **argv) {
    wbArg_t args;
    args.args = argv;
    args.count = argc;
    return args;
}

const char *wbArg_getInputFile(wbArg_t args, int index) {
    for (int i = 1; i < args.count; ++i) {
        if (strcmp(args.args[i], "-i") == 0) {
            // Get the comma-separated list of files
            char *inputFiles = args.args[i + 1];
            char *inputFilesCopy = strdup(inputFiles); // Make a copy of the string
            char *token;
            int currentIndex = 0;
            printf("%s\n", inputFiles);

            // Tokenize the inputFiles string by commas
            token = strtok(inputFilesCopy, ",");
            while (token != NULL) {
                if (currentIndex == index) {
                    printf("Get input file from %s\n", token);
                    return token;
                }
                token = strtok(NULL, ",");
                currentIndex++;
            }
        }
    }
    fprintf(stderr, "No input file found!");
    return NULL;
}


const char *wbArg_getOutputFile(wbArg_t args) {
    for (int i = 1; i < args.count; ++i) {
        if (strcmp(args.args[i], "-e") == 0) {
            return args.args[i + 1];
        }
    }
    return NULL;
}

void formatMessage(char *buffer, size_t bufferSize, const char *format, va_list args) {
    vsnprintf(buffer, bufferSize, format, args);
}

void wbTime_start(const char *label, const char *message, ...) {
    char buffer[256];  // Adjust buffer size as needed
    va_list args;
    va_start(args, message);
    formatMessage(buffer, sizeof(buffer), message, args);
    va_end(args);
    printf("Starting %s: %s\n", label, buffer);
}

void wbTime_stop(const char *label, const char *message, ...) {
    char buffer[256];  // Adjust buffer size as needed
    va_list args;
    va_start(args, message);
    formatMessage(buffer, sizeof(buffer), message, args);
    va_end(args);
    printf("Stopping %s: %s\n", label, buffer);
}

void wbLog(const char *level, const char *message, ...) {
    char buffer[256];  // Adjust buffer size as needed
    va_list args;
    va_start(args, message);
    formatMessage(buffer, sizeof(buffer), message, args);
    va_end(args);
    printf("[%s] %s\n", level, buffer);
}


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


#endif // WB_H
