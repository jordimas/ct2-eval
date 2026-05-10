#!/bin/bash


models=("google/gemma-4-31B")

# Loop through each model
for model in "${models[@]}"; do
    # Create a directory-friendly name
    model_base="${model##*/}"
    dir_name="${model_base//-/_}.ct2"  # Replace '-' with '_' and add _ct2

    echo "Converting model $model into directory $dir_name ..."

    # Run the converter
    ct2-transformers-converter --model "$model" --force --quantization int8 --output_dir "$dir_name"

    echo "Run inference on $model ..."


    # Run the Python script with the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE python gemma4-sample.py --model "$dir_name" --tokenizer "$model"
done

echo "All models processed."
