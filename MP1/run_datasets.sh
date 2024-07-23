#!/bin/bash

mkdir -p bench

for i in 0 1 2 3 4 5 6 7 8 9; do
	echo "--------------";
	echo "Dataset " $i 
	echo "Running: ./template -e ./data/${i}/output.raw -i ./data/${i}/input0.raw,./data/${i}/input1.raw -t vector"
	./template.exe -e ./data/${i}/output.raw -i ./data/${i}/input0.raw,./data/${i}/input1.raw -t vector
done


