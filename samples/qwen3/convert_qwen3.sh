#!/bin/bash

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

# List of models
##models=("Qwen3-4B-Thinking-2507", "Qwen3-4B-Instruct-2507")
#models=("Qwen3-4B-Instruct-2507")
#models=("Qwen3-32B" "Qwen3-14B" "Qwen3-4B" "Qwen3-4B-Thinking-2507", "Qwen3-4B-Instruct-2507")
models=("Qwen3-30B-A3B")


# Loop through each model
for model in "${models[@]}"; do
    # Create a directory-friendly name
    dir_name="${model//-/_}.ct2"  # Replace '-' with '_' and add _ct2

    echo "Converting model $model into directory $dir_name ..."

    # Run the converter
    ct2-transformers-converter --model "Qwen/$model" --force --quantization int8 --output_dir "$dir_name"
    python qwen3.py --hf_model "Qwen/$model"  --ct2_model "${model//-/_}.ct2"

    # Run the Python script with the environment variable
done

echo "All models processed."
