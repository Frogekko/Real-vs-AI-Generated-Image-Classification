# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:55:27 2025

@author: Fredrik

Model trainig
"""
import os
from tqdm import tqdm
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)

num_ftrs = model.fc.in_features
# Changing the last layer to output 2 classes (real and fake)
model.fc = nn.Linear(num_ftrs, 2)


transform = transforms.Compose([
transforms.ToTensor(),  # Converts PIL image to Tensor and scales to [0, 1]
transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                     std=[0.229, 0.224, 0.225])   # ImageNet stds
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# =============================================================================

def training_from_scratch(train):
    # Loading the dataset from their respective image folder.
    train_dataset = datasets.ImageFolder(train, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Printing to enshour that the folder structure is correct
    print(f"Loaded {len(train_dataset)} training data.")
    print(f"Classes found: {train_dataset.classes}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a type of opimization algorithm
    # model.parameters provides all the parameter of the model that should be uploaded during training.
    # lr=1e-4 sets the learning rate to 0.0001
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop (the amount would need some testing)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        
    print("Training finished")
    try:
        save_path = "classifier_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved as {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def testing_model(test_data, saved_model):   
    # Loading the dataset from their respective image folder.
    test_dataset = datasets.ImageFolder(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.load_state_dict(torch.load(saved_model, weights_only=False))
    model.eval()
    
    image_count = 0
    correct = 0
    total = 0
    all_labels =  []
    all_preds = []
    for root, dirs, files in os.walk(test_data):
        image_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Total images: {image_count}")
    
    with torch.no_grad():
        for img, labels in tqdm(test_loader, desc="Testing model", unit="img"):
            img ,labels = img.to(device), labels.to(device)
            
            outputs = model(img)
            _, preds = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    print("\nClassification Report")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    print("\nConfusion Matrix")
    print(confusion_matrix(all_labels, all_preds))