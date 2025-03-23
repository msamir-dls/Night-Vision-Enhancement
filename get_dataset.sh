#!/bin/bash

TARGET_DIR="/Users/samirmishra/Downloads/lowtohighimage/"

# Download the dataset
wget -O "$TARGET_DIR/lol_dataset.zip" https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip

# Unzip the dataset into the target directory and remove the zip file
unzip -q "$TARGET_DIR/lol_dataset.zip" -d "$TARGET_DIR" && rm "$TARGET_DIR/lol_dataset.zip"

echo "Dataset downloaded and extracted to $TARGET_DIR"