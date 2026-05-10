#!/bin/bash

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

# List of models
models=(
    "google/t5gemma-s-s-ul2"
    "google/t5gemma-s-s-prefixlm"
    "google/t5gemma-s-s-ul2-it"
    "google/t5gemma-s-s-prefixlm-it"
    "google/t5gemma-b-b-ul2"
    "google/t5gemma-b-b-prefixlm"
    "google/t5gemma-b-b-ul2-it"
    "google/t5gemma-b-b-prefixlm-it"
    "google/t5gemma-l-l-ul2"
    "google/t5gemma-l-l-prefixlm"
    "google/t5gemma-l-l-ul2-it"
    "google/t5gemma-l-l-prefixlm-it"
    "google/t5gemma-ml-ml-ul2"
    "google/t5gemma-ml-ml-prefixlm"
    "google/t5gemma-ml-ml-ul2-it"
    "google/t5gemma-ml-ml-prefixlm-it"
    "google/t5gemma-xl-xl-ul2"
    "google/t5gemma-xl-xl-prefixlm"
    "google/t5gemma-xl-xl-ul2-it"
    "google/t5gemma-xl-xl-prefixlm-it"
    "google/t5gemma-2b-2b-ul2"
    "google/t5gemma-2b-2b-prefixlm"
    "google/t5gemma-2b-2b-ul2-it"
    "google/t5gemma-2b-2b-prefixlm-it"
    "google/t5gemma-9b-9b-ul2"
    "google/t5gemma-9b-9b-prefixlm"
    "google/t5gemma-9b-9b-ul2-it"
    "google/t5gemma-9b-9b-prefixlm-it"
    "google/t5gemma-9b-2b-ul2"
    "google/t5gemma-9b-2b-prefixlm"
    "google/t5gemma-9b-2b-ul2-it"
    "google/t5gemma-9b-2b-prefixlm-it"
)

models=(
    # Small (0.3B)
    "google/t5gemma-9b-2b-prefixlm-it")

# Loop through each model
for model in "${models[@]}"; do
    # Create a directory-friendly name
    dir_name="${model//-/_}.ct2"  # Replace '-' with '_' and add _ct2

    echo "Converting model $model into directory $dir_name ..."

    # Run the converter
    ct2-transformers-converter --model "$model" --force --quantization int8 --output_dir "$dir_name"

    # Run the Python script with the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE python t5.py --hf-model $model --ct2-model "$dir_name"
done

echo "All models processed."
