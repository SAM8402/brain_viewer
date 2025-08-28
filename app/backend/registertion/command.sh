#!/bin/bash

# BFI Viewer Registration Pipeline
# Processes brain slice registration using local BFI and Nissl images

set -e  # Exit on any error

echo "Starting BFI Viewer Registration Pipeline..."
echo "=========================================="
pip install -r requirements.txt
echo "Step 1: Generating stack-to-BFW transformations..."
python3 first.py

echo "Step 2: Processing local BFI to Nissl registration..."
python3 second.py

echo "Step 3: Validating transformation completeness..."
python3 third.py ./142_t_to_s.json

echo "Step 4: Converting thumbnail-to-canvas transformations..."
python3 forth.py ./142_t_to_s.json ./142_c_to_s.json --factor 64
# python3 forth.py ./142_t_to_s.json ./142_c_to_s.json 

echo "Step 5: Chaining final transformations..."
python3 fifth.py ./142_c_to_s.json ./all_slice_transformations.json ./142_original_to_bfiw_final.json --sample_id 142

echo "=========================================="
echo "Registration pipeline completed successfully!"
echo "Final transformation file: ./142_original_to_bfiw_final.json"