#!/bin/bash

nvcc template.cu -o template

./template -e ./data/1/output.dat -i ./data/1/input.dat,./data/1/kernel.dat -t vector

# ./test -i ./data/1/test.dat