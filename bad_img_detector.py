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

bad_img_lsit = [] # Stores bad images

def bad_images(file_path):
    
    warnings.simplefilter('ignore', Image.DecompressionBombWarning) # If you dont trust the dataset you may whant to hash out this.
    Image.MAX_IMAGE_PIXELS = None # Sets the maximum number of pixelse to unlimited
    
    # Get list of image files
    image_files = [f for f in listdir(file_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Checking images", unit="image"):
        full_path = path.join(file_path, filename)
        try:
            img = Image.open(full_path) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError):
            print('Bad file:', filename) # print out the names of corrupt files
            bad_img_lsit.append(filename)
    
    with open("list_of_bad_images.txt", "w") as file:
        for img in bad_img_lsit:
            file.write(file_path+' '+img+'\n')
        file.close()