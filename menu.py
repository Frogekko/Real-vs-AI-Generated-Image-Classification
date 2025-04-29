# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:10:12 2025

@author: Fredrik
"""
from preprocessing import resizer


# input_folder and output_folder will be the directory containing the images to resize
def prepmain():
    print("Hello, you can preprocess you'r dataset of images here")
    print("Would you like to look for 'bad images' or 'resize'? ")
    choice = input().lower()
    while choice == "resize":
        print("1. prep train real")
        print("2. prep train fake")
        print("3. prep test real")
        print("4. prep test fake")
        print("5. end task")
        choice = int(input())
        
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
        elif choice == 5:
            break
        else:
            print("Please enter a valid choice")
            
    while choice == "bad images":
        print("1. bad images train real")
        print("2. bad images train fake")
        print("3. bad images test real")
        print("4. bad images test fake")
        print("5. end task")
        choice = int(input())
        
        
prepmain()