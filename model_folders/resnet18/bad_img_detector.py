# -*- coding: utf-8 -*-
"""
@author: Group 20

Source: https://opensource.com/article/17/2/python-tricks-artists
"""

from os import listdir,path
from PIL import Image
import warnings
from tqdm import tqdm

bad_img_lsit = [] # Stores bad images

def bad_images(file_path):
    
    warnings.simplefilter('ignore', Image.DecompressionBombWarning) # If you dont trust the dataset you may whant to hash out this.
    Image.MAX_IMAGE_PIXELS = None # Sets the maximum number of pixelse to unlimited
    
    # Get list of image files with full paths
    image_files = [path.join(file_path, f) for f in listdir(file_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Checking images", unit="image"):
        full_path = path.join(file_path, filename)
        try:
            with Image.open(full_path) as img:
                img.verify() # verify that it is, in fact an image
            with Image.open(full_path) as img:
                img.load() # This is to forcefully load the image to catch the images that verify does not catch
        except (IOError, SyntaxError) as e:
            print(f"Bad file: {full_path} — {e }") # print out the names of corrupt files
            bad_img_lsit.append(full_path)
    
    with open("list_of_bad_images.txt", "w") as file:
        for img in bad_img_lsit:
            file.write(img+'\n')