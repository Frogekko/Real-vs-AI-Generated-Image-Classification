# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 23:14:19 2025

@author: Fredrik
"""
import os
from PIL import Image
from tqdm import tqdm

# input_folder and output_folder will be the directory containing the images to resize
def prepmain():
    print("Hello, you can preprocess you'r dataset of images here")
    print("1. prep train real")
    print("2. prep train fake")
    print("3. prep test real")
    print("4. prep test fake")
    choice = input(int())
    
    if choice == 1:
        input_folder = 'C:/Users/Fredrik/MLEksamen/train/real'
        output_folder = 'C:/Users/Fredrik/MLEksamen/resized_train/resized_real'
        resizer(input_folder, output_folder)
    elif choice == 2:
        input_folder ='C:/Users/Fredrik/MLEksamen/train/fake'
        output_folder = 'C:/Users/Fredrik/MLEksamen/resized_train/resized_fake'
        resizer(input_folder, output_folder)
    elif choice == 3:
        input_folder = 'C:/Users/Fredrik/MLEksamen/test/real'
        output_folder = 'C:/Users/Fredrik/MLEksamen/resized_test/resized_real'
        resizer(input_folder, output_folder)
    elif choice == 4:
        input_folder = 'C:/Users/Fredrik/MLEksamen/test/fake'
        output_folder = 'C:/Users/Fredrik/MLEksamen/resized_test/resized_fake'
        resizer(input_folder, output_folder)
    else:
        print("Please enter a valid choice")
        prepmain()
    
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
    
        # Open the image
        with Image.open(img_path) as img:
            # Convert the image to RGB if it's in the palette-based mode(P)
            if img.mode == 'P':
                img = img.convert('RGB')
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

prepmain()
