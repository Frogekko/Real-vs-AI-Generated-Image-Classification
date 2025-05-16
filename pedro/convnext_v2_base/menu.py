# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:10:12 2025

@author: Group 20

Remember to change the input_folder, output_folder and file_path so it fits to where you have saved the dataset
"""
from preprocessing import resizer
from bad_img_detector import bad_images, bad_img_lsit
from model_training_testing import train_model, testing_testdataset, testing_traindataset

# input_folder and output_folder will be the directory containing the images to resize
def prepmain():
    print("Hello, welcome to the image preprocessing and model training menu")
    while True:
        print("\nMain Menu:")
        print("1. Look for bad images")
        print("2. Resize images from the dataset")
        print("3. Train/Test the model")
        print("0. Exit")
        choice = input("Enter your choice as a number: ")
        
        if choice == "1":
            while True:
                print("\nBad Images Menu:")
                print("1. Bad images train real")
                print("2. Bad images train fake")
                print("3. Bad images test real")
                print("4. Bad images test fake")
                print("5. List of bad images")
                print("0. Exit")
                sec_choice = input("Enter your choice as a number: ")

                if sec_choice == "1":
                    input_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train/real'
                    output_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_train/resized_real'
                    resizer(input_folder, output_folder)
                elif sec_choice == "2":
                    input_folder ='C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train/fake'
                    output_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_train/resized_fake'
                    resizer(input_folder, output_folder)
                elif sec_choice == "3":
                    input_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test/real'
                    output_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_test/resized_real'
                    resizer(input_folder, output_folder)
                elif sec_choice == "4":
                    input_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test/fake'
                    output_folder = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_test/resized_fake'
                    resizer(input_folder, output_folder)
                elif sec_choice == "0":
                    break
                else:
                    print("Please enter a valid choice")

        elif choice == "2":
            while True:
                print("1. Resize train real")
                print("2. Resize train fake")
                print("3. Resize test real")
                print("4. Resize test fake")
                print("0. Exit")
                sec_choice = input("Enter your choice as a number: ")
                
                if sec_choice == "1":
                    file_path = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train/real'
                    bad_images(file_path)
                elif sec_choice == "2":
                    file_path ='C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train/fake'
                    bad_images(file_path)
                elif sec_choice == "3":
                    file_path = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test/real'
                    bad_images(file_path)
                elif sec_choice == "4":
                    file_path = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test/fake'
                    bad_images(file_path)
                elif sec_choice == "5":
                    print(bad_img_lsit)
                elif sec_choice == "0":
                    break
                else:
                    print("Please enter a valid choice")
                
        elif choice == "3":
            while True:
                print("\nModel Traning Menu")
                print("1. Train the model")
                print("2. Use the model on the train dataset")
                print("3. Use the model on the test dataset")
                print("0. Exit")
                sec_choice = input("Enter your choice as a number: ")
                if sec_choice == "1":
                    train = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_train'
                    train_model(train)
                elif sec_choice == "2":
                    train = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_train'
                    saved_model = 'C:/Users/p2fre/03 - University/03 - Class Repositories/eksamen_ml/Real-vs-AI-Generated-Image-Classification/pedro/convnext_v2_base/classifier_model.pth'
                    testing_testdataset(train, saved_model)
                elif sec_choice == "3":
                    test_data = 'C:/Users/p2fre/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/resized_test'
                    saved_model = 'C:/Users/p2fre/03 - University/03 - Class Repositories/eksamen_ml/Real-vs-AI-Generated-Image-Classification/pedro/convnext_v2_base/classifier_model.pth'
                    testing_traindataset(test_data, saved_model)
                elif sec_choice == "0":
                    break
        elif choice == "0":
            break
        
prepmain()