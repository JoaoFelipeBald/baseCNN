import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch

NUM_WORKERS = os.cpu_count()
'''
def create_dataloaders_single_dir(
    data_dir: str,
    transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    train_portion: float = 0.8,  # Fraction of data to use for training
    val_size: float = 0.2,        # Fraction of training data to use for validation
    data_portion: float = 1.0      # Fraction of the dataset to use for DataLoaders
):
    """Creates training, validation, and testing DataLoaders from a single data directory.

    Args:
        data_dir: Path to the data directory containing all images.
        transform: torchvision transforms to perform on training and validation data.
        test_transform: torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        train_portion: Fraction of data to use for training.
        val_size: Fraction of training data to use for validation.
        data_portion: Fraction of the entire dataset to use for DataLoaders.

    Returns:
        A tuple of (train_dataloader, val_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    # Load the dataset without any transform initially
    full_data = datasets.ImageFolder(data_dir)

    # Get class names
    class_names = full_data.classes

    # Calculate the number of samples for train, val, and test
    num_samples = int(len(full_data) * data_portion)
    num_train_samples = int(num_samples * train_portion)
    num_val_samples = int(num_train_samples * val_size)
    num_train_samples -= num_val_samples  # Adjust training size
    num_test_samples = num_samples - num_train_samples - num_val_samples

    # Randomly sample indices for train, val, and test data
    indices = torch.randperm(len(full_data))[:num_samples]  # Randomly sample indices
    sampled_data = Subset(full_data, indices.tolist())  # Create a subset with the sampled indices

    # Split the dataset into train, val, and test subsets
    train_subset, val_subset, test_subset = random_split(
        sampled_data,
        [num_train_samples, num_val_samples, num_test_samples]
    )

    # Wrap the subsets in DataLoader, applying the correct transformations for each split
    # Apply the training transform to the training subset
    train_data = Subset(datasets.ImageFolder(data_dir, transform=transform), train_subset.indices)

    # Apply the test transform to validation and test subsets
    val_data = Subset(datasets.ImageFolder(data_dir, transform=test_transform), val_subset.indices)
    test_data = Subset(datasets.ImageFolder(data_dir, transform=test_transform), test_subset.indices)

    # Create DataLoaders for each subset
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch

NUM_WORKERS = os.cpu_count()

def create_dataloaders_single_dir(
    data_dir: str,
    transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    train_portion: float = 0.8,
    val_size: float = 0.2,
    data_portion: float = 1.0
):
    full_data = datasets.ImageFolder(data_dir)
    class_names = full_data.classes

    # Calculate samples for train, val, and test
    num_samples = int(len(full_data) * data_portion)
    num_train_samples = int(num_samples * train_portion)
    num_val_samples = int(num_train_samples * val_size)
    num_train_samples -= num_val_samples
    num_test_samples = num_samples - num_train_samples - num_val_samples

    # Sample indices
    indices = torch.randperm(len(full_data))[:num_samples]
    train_indices, val_indices, test_indices = random_split(
        indices.tolist(),
        [num_train_samples, num_val_samples, num_test_samples]
    )

    # Create transformed subsets
    train_data = Subset(full_data, train_indices)
    train_data.dataset.transform = transform
    val_data = Subset(full_data, val_indices)
    val_data.dataset.transform = test_transform
    test_data = Subset(full_data, test_indices)
    test_data.dataset.transform = test_transform

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names



import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

NUM_WORKERS = os.cpu_count()

def create_partial_dataloaders2(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    train_portion: float = 1.0,  # Fraction of training data to use
    test_portion: float = 1.0,    # Fraction of testing data to use
    val_size: float = 0.2,        # Fraction of training data to use for validation
):
    """Creates training, validation, and testing DataLoaders with a portion of the dataset.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders with a portion
    of the data if specified.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training data.
        test_transform: torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        train_portion: Fraction of training data to use.
        test_portion: Fraction of testing data to use.
        val_size: Fraction of training data to use for validation.

    Returns:
        A tuple of (train_dataloader, val_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    # Determine the number of samples for training and validation
    num_train_samples = int(len(train_data) * train_portion)
    num_val_samples = int(num_train_samples * val_size)
    num_train_samples -= num_val_samples  # Adjust training size

    # Randomly sample indices for train and val data
    train_subset, val_subset = random_split(train_data, [num_train_samples, num_val_samples])

    # Determine the number of samples for testing
    num_test_samples = int(len(test_data) * test_portion)
    test_subset = random_split(test_data, [num_test_samples, len(test_data) - num_test_samples])[0]

    # Create DataLoaders for each subset
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names

def a():
    print("a")
