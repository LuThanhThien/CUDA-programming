// wb.h

#ifndef WB_H
#define WB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>

#define ERROR "ERROR"
#define TRACE "TRACE"
#define GPU "GPU"
#define CPU "CPU"
#define Generic "Generic"
#define Compute "Compute"
#define Copy "Copy"

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


void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA error after %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


#endif // WB_H
