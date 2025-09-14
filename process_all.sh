#!/bin/bash

# This script processes image folders using the Retinexformer model.
# It takes a single argument: the main directory containing the subfolders to process.

# --- Configuration ---
# Set the model you want to use here.
# Make sure the WEIGHTS and CONFIG files match.
WEIGHTS=("pretrained_weights/LOL_v2_real.pth" "pretrained_weights/LOL_v2_synthetic.pth")
CONFIGS=("Options/Retinexformer_LOL_v2_real.yml" "Options/Retinexformer_LOL_v2_synthetic.yml")
NAMES=("retinexformer_LOL_v2_real" "retinexformer_LOL_v2_synthetic")
# ---------------------


# Check if the user has provided a directory
if [ -z "$1" ]; then
  echo "Usage: $0 <main_directory>"
  exit 1
fi

MAIN_DIR=$1
PYTHON_SCRIPT="process_folder.py"

# Find all subdirectories in the main directory and process them
find "$MAIN_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r SUB_DIR; do
  echo "--- Processing subdirectory: $SUB_DIR ---"

  # Define the folders to process within each subdirectory
  TARGET_FOLDERS=("images_8" "images_8_contrast" "images_8_multiexposure" "images_8_variance")

  for i in "${!WEIGHTS[@]}"; do
    WEIGHT="${WEIGHTS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    echo "Using weights: $WEIGHT"
    echo "Using config: $CONFIG"
    echo "Output name suffix: $NAME"

    for FOLDER in "${TARGET_FOLDERS[@]}"; do
      INPUT_FOLDER="$SUB_DIR/$FOLDER"
      OUTPUT_FOLDER="${INPUT_FOLDER}_${NAME}"

      if [ -d "$INPUT_FOLDER" ]; then
        echo "Processing folder: $INPUT_FOLDER"
        python "$PYTHON_SCRIPT" --opt "$CONFIG" --weights "$WEIGHT" --input_folder "$INPUT_FOLDER" --output_folder "$OUTPUT_FOLDER"
      else
        echo "Skipping... Folder not found: $INPUT_FOLDER"
      fi
    done
  done
done

echo "--- All processing complete. ---"