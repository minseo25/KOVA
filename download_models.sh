#!/bin/bash

mkdir -p models

echo "Downloading gemma-2-2b-it..."
huggingface-cli download google/gemma-2-2b-it --local-dir="models/gemma-2-2b-it"

echo "Downloading llama-3-Korean-Bllossom-8B-Q4_K_M..."
huggingface-cli download MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M --local-dir="models/llama-3-korean-bllossom-8b-gguf"

echo "Downloading pyannote/segmentation-3.0..."
huggingface-cli download pyannote/segmentation-3.0 --local-dir="models/pyannote-segmentation-3"

echo "Downloading faster-whisper-large-v3..."
huggingface-cli download Systran/faster-whisper-small --local-dir="models/faster-whisper-small"

echo "Downloads completed."
