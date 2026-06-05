#!/bin/bash


models=("google/gemma-4-31b-it" "google/gemma-4-12b-it")

# Loop through each model
for model in "${models[@]}"; do
    echo "Converting model $model ..."

    # Run the converter — output dir mirrors the HF model ID (google/modelname)
    ct2-transformers-converter --low_cpu_mem_usage --model "$model" --force --quantization int8 --output_dir "models/$model"

    echo "Run inference on $model ..."

    # Run the Python script with the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE python gemma4-sample.py --model "models/$model" --tokenizer "$model"
done

echo "All models processed."
