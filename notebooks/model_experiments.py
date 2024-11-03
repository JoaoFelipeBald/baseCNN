import os
current_directory = os.getcwd()

last_name = os.path.basename(current_directory)
if last_name=="notebooks":
    os.chdir('..')

import random
import torch
# Set the seed for general torch operations
torch.manual_seed(42)
# Set the seed for CUDA torch operations (ones that happen on the GPU)
torch.cuda.manual_seed(42)
random.seed(42)
sweep_config = {
    'method': 'random',  # Use 'random' instead of 'grid'
    'metric': {'name': 'validation_loss', 'goal': 'minimize'},  # Metric to optimize
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',  # Use uniform distribution for random sampling
            'min': 1e-3,
            'max': 1e-2
        },
        'batch_size': {
            'values': [32]  # Fixed batch size
        },
        'epochs': {
            'value': 60  # Fixed value for all runs
        },
        'patience': {
            'distribution': 'int_uniform',  # Random integers between min and max
            'min': 6,
            'max': 7
        },
        'patience_step': {
            'distribution': 'int_uniform',  # Random integers between min and max
            'min': 3,
            'max': 4
        },
        'device': {
            'value': 'cuda' if torch.cuda.is_available() else 'cpu'  # Fixed value based on availability
        }
    }
}

import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path

import requests



'''
%pip install torchinfo
%pip install -q path
%pip install wandb
%pip install -q kaggle
%pip install kaggle
%pip install torch_lr_finder
%pip install torch
%pip install torchinfo
%pip install -q path
%pip install wandb
%pip install -q kaggle
%pip install torchtext
%pip install torchvision
'''



import torchvision
from torchvision import datasets, transforms

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
auto_transforms = weights.transforms()

augmentation_transforms = transforms.Compose([
#    transforms.RandomHorizontalFlip(),                      # Randomly flip images horizontally
#    transforms.RandomRotation(20),                          # Randomly rotate images by up to 20 degrees
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
#    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
#    transforms.RandomGrayscale(p=1),                      # Convert some images to grayscale with a probability of 10%
#    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop to 224x224
#    transforms.RandomVerticalFlip(),                         # Randomly flip images vertically
#    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet mean and std
    transforms.ToTensor(),                                   # Convert images to tensor
    auto_transforms,                                          # Apply pretrained model's transforms (normalization)
    transforms.Resize((224, 224))
])


import importlib
from src.dataloaders import data
importlib.reload(data)  # Reload the module to apply any changes
from src.dataloaders.data import create_dataloaders_single_dir


device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # Use default pre-trained weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Freeze the feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False
# Define the classifier for your specific task
output_shape = 2
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=output_shape, bias=True)
).to(device)


data_dir=""

train_dataloader, val_loader, test_dataloader, class_names = create_dataloaders_single_dir(
    data_dir=data_dir,
    transform=augmentation_transforms,
    test_transform=auto_transforms,
    batch_size=32,
    train_portion=0.85,
    val_size=0.1,
    data_portion=0.3)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#find_lr(optimizer=optimizer,model=model,train_dataloader=train_dataloader,loss_fn=loss_fn,device=device,end_lr=1,num_iter=100)

for images, labels in train_dataloader:
    # labels contains the indices of the classes for each image in the batch
    class_labels = [class_names[label] for label in labels]
    
    # Print the class of each image in the batch
    for i, class_label in enumerate(class_labels):
        print(f"Image {i} class: {class_label}")
    
    # Break after one batch if you only need to check a few images
    break

import importlib
from src.training import traincnn  # Import the module first
importlib.reload(traincnn)  # Reload the module to apply any changes
from src.training.traincnn import train_class
train=train_class(model=model,config=sweep_config,
              loss=loss_fn,device=device,
              checkpoints="checkpoints",train_loader=train_dataloader,
              test_loader=test_dataloader, project_name="git1",step_factor=0.2,
              optimizer=torch.optim.Adam(model.parameters(), 0), max_norm=1.4)
train.train_model(1)
