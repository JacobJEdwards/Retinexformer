#!/bin/bash

WEIGHTS=(
  "pretrained_weights/LOL_v1.pth"
  "pretrained_weights/LOL_v2_real.pth"
  "pretrained_weights/LOL_v2_synthetic.pth"
  "pretrained_weights/NTIRE.pth"
  "pretrained_weights/SDSD_indoor.pth"
  "pretrained_weights/SDSD_outdoor.pth"
  "pretrained_weights/SID.pth"
  "pretrained_weights/SMID.pth"
  "pretrained_weights/FiveK.pth"
  "pretrained_weights/MST_Plus_Plus_NTIRE/MST_Plus_Plus_4x1800.pth"
  "pretrained_weights/MST_Plus_Plus_NTIRE/MST_Plus_Plus_8x1150.pth"
)

CONFIGS=(
  "Options/RetinexFormer_LOL_v1.yml"
  "Options/RetinexFormer_LOL_v2_real.yml"
  "Options/RetinexFormer_LOL_v2_synthetic.yml"
  "Options/RetinexFormer_NTIRE.yml"
  "Options/RetinexFormer_SDSD_indoor.yml"
  "Options/RetinexFormer_SDSD_outdoor.yml"
  "Options/RetinexFormer_SID.yml"
  "Options/RetinexFormer_SMID.yml"
  "Options/RetinexFormer_FiveK.yml"
  "Options/MST_Plus_Plus_NTIRE_4x1800.yml"
  "Options/MST_Plus_Plus_NTIRE_8x1150.yml"
)
NAMES=(
  "retinexformer_LOL_v1"
  "retinexformer_LOL_v2_real"
  "retinexformer_LOL_v2_synthetic"
  "retinexformer_NTIRE"
  "retinexformer_SDSD_indoor"
  "retinexformer_SDSD_outdoor"
  "retinexformer_SID"
  "retinexformer_SMID"
  "retinexformer_FiveK"
  "retinexformer_MST_Plus_Plus_4x1800"
  "retinexformer_MST_Plus_Plus_8x1150"
)


if [ -z "$1" ]; then
  echo "Usage: $0 <main_directory>"
  exit 1
fi

MAIN_DIR=$1
PYTHON_SCRIPT="process_folder.py"

find "$MAIN_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r SUB_DIR; do
  echo "--- Processing subdirectory: $SUB_DIR ---"

  TARGET_FOLDERS=("images")

  for i in "${!WEIGHTS[@]}"; do
    WEIGHT="${WEIGHTS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    echo "Using weights: $WEIGHT"
    echo "Using config: $CONFIG"
    echo "Output name suffix: $NAME"

    for FOLDER in "${TARGET_FOLDERS[@]}"; do
      INPUT_FOLDER="$SUB_DIR/$FOLDER"
      OUTPUT_FOLDER="$SUB_DIR/retinex/$NAME"

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