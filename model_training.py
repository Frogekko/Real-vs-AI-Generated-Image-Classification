# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:55:27 2025

@author: Fredrik

Model trainig
"""
import os
import torch
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset


def training_from_scratch(train, test):
    transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to Tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])
    
    # Loading the dataset from their respective image folder.
    train_dataset = datasets.ImageFolder(train, transform=transform)
    test_dataset = datasets.ImageFolder(test, transform=transform)
    
    
    # Assigning labels for real and fake
    for dataset, label in [(train_dataset, 1), (test_dataset, 0)]:
        dataset.targets = [label] * len(dataset)
        
    # I will be needing to combine the train sets and test sets sinse the model needs to train on both classes during each epoch so it can learn the paterns and the differences between them.
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a type of opimization algorithm
    # model.parameters provides all the parameter of the model that should be uploaded during training.
    # lr=1e-4 sets the learning rate to 0.0001
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop (the amount would need some testing)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.4f}")
        
    print("Training finished")