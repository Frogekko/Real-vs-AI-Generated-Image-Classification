# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:10:12 2025

@author: Fredrik
"""
from preprocessing import resizer
from bad_img_detector import bad_images

# input_folder and output_folder will be the directory containing the images to resize
def prepmain():
    print("Hello, you can preprocess you'r dataset of images here")
    while True:
        print("Would you like to look for 'bad images' or 'resize'? ")
        print("\nMain Menu:")
        print("1. Look for bad images")
        print("2. Resize images from the dataset")
        print("0. Exit")
        choice = input("Enter your choice as int: ")
        if choice == "2":
            while True:
                print("1. prep train real")
                print("2. prep train fake")
                print("3. prep test real")
                print("4. prep test fake")
                print("5. end task")
                sec_choice = input("Enter your choice as int: ")
                
                if sec_choice == "1":
                    input_folder = 'C:/Users/Fredrik/MLEksamen/train/real'
                    output_folder = 'C:/Users/Fredrik/MLEksamen/resized_train/resized_real'
                    resizer(input_folder, output_folder)
                elif sec_choice == "2":
                    input_folder ='C:/Users/Fredrik/MLEksamen/train/fake'
                    output_folder = 'C:/Users/Fredrik/MLEksamen/resized_train/resized_fake'
                    resizer(input_folder, output_folder)
                elif sec_choice == "3":
                    input_folder = 'C:/Users/Fredrik/MLEksamen/test/real'
                    output_folder = 'C:/Users/Fredrik/MLEksamen/resized_test/resized_real'
                    resizer(input_folder, output_folder)
                elif sec_choice == "4":
                    input_folder = 'C:/Users/Fredrik/MLEksamen/test/fake'
                    output_folder = 'C:/Users/Fredrik/MLEksamen/resized_test/resized_fake'
                    resizer(input_folder, output_folder)
                elif sec_choice == "5":
                    break
                else:
                    print("Please enter a valid choice")
                
        elif choice == "1":
            while True:
                print("\nBad Images Menu:")
                print("1. bad images train real")
                print("2. bad images train fake")
                print("3. bad images test real")
                print("4. bad images test fake")
                print("5. end task")
                sec_choice = input("Enter your choice as int: ")
                
                if sec_choice == "1":
                    file_path = 'C:/Users/Fredrik/MLEksamen/train/real'
                    bad_images(file_path)
                elif sec_choice == "2":
                    file_path ='C:/Users/Fredrik/MLEksamen/train/fake'
                    bad_images(file_path)
                elif sec_choice == "3":
                    file_path = 'C:/Users/Fredrik/MLEksamen/test/real'
                    bad_images(file_path)
                elif sec_choice == "4":
                    file_path = 'C:/Users/Fredrik/MLEksamen/test/fake'
                    bad_images(file_path)
                elif sec_choice == "5":
                    break
                else:
                    print("Please enter a valid choice")
        elif choice == "0":
            break
        
prepmain()