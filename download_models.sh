#!/bin/bash

mkdir -p models

echo "Downloading gemma-2-2b-it..."
huggingface-cli download google/gemma-2-2b-it --local-dir="models/gemma-2-2b-it"

echo "Downloading llama-3-Korean-Bllossom-8B-Q4_K_M..."
huggingface-cli download MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M --local-dir="models/llama-3-korean-bllossom-8b-gguf"

echo "Downloading qwen2-7b-instruct-q5_k_m.gguf..."
huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF qwen2-7b-instruct-q5_k_m.gguf --local-dir="models/qwen2-7b-instruct-gguf" --local-dir-use-symlinks False

echo "Downloading faster-whisper-large-v3..."
huggingface-cli download Systran/faster-whisper-small --local-dir="models/faster-whisper-small"

echo "Downloads completed."
