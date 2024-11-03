import torchvision
from torchvision import datasets, transforms

weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
auto_transforms = weights.transforms()

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),                      # Randomly flip images horizontally
    transforms.RandomRotation(20),                          # Randomly rotate images by up to 20 degrees
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
#    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
#    transforms.RandomGrayscale(p=1),                      # Convert some images to grayscale with a probability of 10%
#    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop to 224x224
#    transforms.RandomVerticalFlip(),                         # Randomly flip images vertically
#    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet mean and std
    transforms.ToTensor(),                                   # Convert images to tensor
    auto_transforms                                          # Apply pretrained model's transforms (normalization)
])
