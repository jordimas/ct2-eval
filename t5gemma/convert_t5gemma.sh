#!/bin/bash

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

# List of models
models=(
    # Small (0.3B)
    "google/t5gemma-s-s-ul2"
    "google/t5gemma-s-s-prefixlm"
    "google/t5gemma-s-s-ul2-it"
    "google/t5gemma-s-s-prefixlm-it")

# Loop through each model
for model in "${models[@]}"; do
    # Create a directory-friendly name
    dir_name="${model//-/_}.ct2"  # Replace '-' with '_' and add _ct2

    echo "Converting model $model into directory $dir_name ..."

    # Run the converter
    ct2-transformers-converter --model "$model" --force --quantization int8 --output_dir "$dir_name"

    # Run the Python script with the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE python t5.py --hf_model $model --ct_model "$dir_name"
done

echo "All models processed."
