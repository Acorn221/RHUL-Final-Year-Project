import os

"""
This script converts the .gif files to .png files in the processed_scans folder.

"""

path = "./processed_scans"

files = os.listdir(path)

for file in files:
  # Rename the file to clip the last 30 characters
  full_path = os.path.join(path, file)

  new_name = file[:-42] + ".gif"
  
  # Rename the file
  os.rename(full_path, os.path.join(path, new_name))


# Convert the .gif files to .png
from PIL import Image

# Loop through all files in the directory
for filename in os.listdir(path):
    if filename.endswith(".gif"):
        # Open the .gif file
        gif_file = Image.open(os.path.join(path, filename))

        # Convert to .png format
        png_file = gif_file.convert('RGB')

        # Save the new .png file
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_file.save(os.path.join(path, png_filename))