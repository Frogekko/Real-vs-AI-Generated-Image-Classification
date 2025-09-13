# -*- coding: utf-8 -*-
"""
@author: Group 20
Script to resize images in a folder to a specified size.
"""
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
    
# Preprocessing
#===========================================================
# Resizeing images

# Loop through all files in the input folder
def resizer(input_folder, output_folder):
    
    # Specify new dimensions (width, height) or calculate based on a factor
    new_width = 224
    new_height = 224
    
    # Get a list of all image files in the input folder
    image_files = [filename for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        img_path = os.path.join(input_folder, filename)
        # Check if the file is an image (you can add more extensions if needed)
        try:
            # Open the image
            with Image.open(img_path) as img:
                # Convert the image to RGB if it's in the palette-based mode(P)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                # Resize the image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipped corrupted file: {filename}-{e}")
