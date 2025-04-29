# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 01:12:47 2025

@author: Fredrik

Source: https://opensource.com/article/17/2/python-tricks-artists
"""

from os import listdir,path
from PIL import Image
import warnings
from tqdm import tqdm

warnings.simplefilter('ignore', Image.DecompressionBombWarning) # If you dont trust the dataset you may whant to hash out this.
Image.MAX_IMAGE_PIXELS = None # Sets the maximum number of pixelse to unlimited

bad_images = [] # Stores bad images
 
user_input = input("Please enter the choosen file path:\n") # User input
file_path = user_input # Variable to remember the user input

# Get list of image files
image_files = [f for f in listdir(file_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(image_files, desc="Checking images", unit="image"):
    full_path = path.join(file_path, filename)
    try:
        img = Image.open(full_path) # open the image file
        img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError):
        print('Bad file:', filename) # print out the names of corrupt files
        bad_images.append(filename)

with open(f"list_of_bad_images.txt", "w") as file:
    for img in bad_images:
        file.write(img + '\n')
    file.close()