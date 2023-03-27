#!/bin/bash

# Define the zip file and target folder

zip_file="Resized_200x200_MIX_2Classes.zip"

target_folder="${zip_file%.zip}"

# Create the target folder if it does not exist

mkdir -p "$target_folder"

# Extract the zip file into the target folder

unzip -q "$zip_file" -d "$target_folder"

echo "Extraction completed."