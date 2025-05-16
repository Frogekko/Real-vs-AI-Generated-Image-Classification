# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:55:27 2025

@author: Group 20

Model training and testing script.
"""
import os
from tqdm import tqdm
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# =============================================================================
def train_model(train):
    val_ratio = 0.1
    # Loading the dataset from their respective image folder.
    train_dataset = datasets.ImageFolder(train, transform=transform)
    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # The dataset is split into train and validation sets
    torch.manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Printing to ensure that the folder structure is correct
    print(f"Loaded {len(train_dataset)} training data.")
    print(f"Loaded {len(val_dataset)} validation data.")
    print(f"Classes found: {train_dataset.dataset.classes}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # Early stopping parameters
    patience = 3  # Number of epochs to wait before stopping if no improvement
    best_val_loss = float('inf')  # Initialize best validation loss
    epochs_no_improve = 0  # Counter for epochs without improvement
    save_path = "classifier_model.pth"  # Path to save the best model
    
    # Training loop
    for epoch in range(50):  # Increased max epochs, early stopping will handle termination
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="img"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate training loss
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}", unit="images"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the model
            try:
                torch.save(model.state_dict(), save_path)
                print(f"Model saved as {save_path} with validation loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        
        # Check if early stopping is triggered
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    print("Training finished")

# =============================================================================
def testing_traindataset(train, saved_model):
    train_dataset = datasets.ImageFolder(train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    model.load_state_dict(torch.load(saved_model, weights_only=False))
    model.eval()
    
    image_count = 0
    correct = 0
    total = 0
    all_labels =  []
    all_preds = []
    for root, dirs, files in os.walk(train):
        image_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Total images: {image_count}")
    
    with torch.no_grad():
        for img, labels in tqdm(train_loader, desc="Testing model", unit="img"):
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
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    
    print("\nConfusion Matrix")
    print(confusion_matrix(all_labels, all_preds))

# =============================================================================
def testing_testdataset(test_data, saved_model):   
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

# =============================================================================