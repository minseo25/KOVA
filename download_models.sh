#!/bin/bash

mkdir -p models

echo "Downloading qwen2.5-1.5b-instruct..."
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir="models/qwen2.5-1.5b"

echo "Downloading pyannote/segmentation-3.0..."
huggingface-cli download pyannote/segmentation-3.0 --local-dir="models/pyannote-segmentation-3"

echo "Downloading pyannote/segmentation..."
huggingface-cli download pyannote/segmentation --local-dir="models/pyannote-segmentation"

echo "Downloads completed."
