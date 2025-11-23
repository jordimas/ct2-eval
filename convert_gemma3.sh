#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# List of models
models=("gemma-3-27b-it" "gemma-3-12b-it" "gemma-3-4b-it" "gemma-3-1b-it")

# Loop through each model
for model in "${models[@]}"; do
    # Create a directory-friendly name
    dir_name="${model//-/_}.ct2"  # Replace '-' with '_' and add _ct2

    echo "Converting model $model into directory $dir_name ..."

    # Run the converter
    ct2-transformers-converter --model "google/$model" --force --quantization int8 --output_dir "$dir_name"

    # Run the Python script with the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE python gemma3.py
done

echo "All models processed."
