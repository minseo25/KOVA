#!/bin/bash

mkdir -p models

echo "Running download_models.py..."
python etc/download_models.py

echo "Downloading llama-3-Korean-Bllossom-8B-Q4_K_M..."
huggingface-cli download MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M --local-dir="models/llama-3-korean-bllossom-8b-gguf"

echo "Downloading qwen2-7b-instruct-q5_k_m.gguf..."
huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF qwen2-7b-instruct-q5_k_m.gguf --local-dir="models/qwen2-7b-instruct-gguf" --local-dir-use-symlinks False

echo "Downloads completed."
