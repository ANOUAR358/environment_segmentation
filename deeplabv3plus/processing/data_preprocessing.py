import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        self.transform = transform
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert mask to tensor without normalization
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# Separate transforms for images and masks
image_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = Compose([
    Resize((512, 512)),
    ToTensor()  # No normalization for masks!
])

# Create datasets with correct transforms
train_dataset = SegmentationDataset(
    images_dir="data/train/images",
    masks_dir="data/train/masks",
    transform=image_transform,
    mask_transform=mask_transform  # Pass separate mask transform
)
    
val_dataset = SegmentationDataset(
    images_dir="data/val/images",
    masks_dir="data/val/masks",
    transform=image_transform,
    mask_transform=mask_transform  # Pass separate mask transform
)
    

from torch.utils.data import DataLoader

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


##second data preprocessing snippet

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

# Define the Segmentation Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes=34, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to one-hot encoded format
        mask_tensor = torch.zeros((self.num_classes, mask.shape[0], mask.shape[1]))
        for class_idx in range(self.num_classes):
            mask_tensor[class_idx] = (mask == class_idx).float()
            
        return image, mask_tensor

# Data Augmentation
def get_training_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=45,
            shift_limit=0.2,
            border_mode=0,
            p=0.5
        ),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.3),
        A.Resize(512, 512),  # Fixed size for batch processing
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512),  # Same size as training for consistency
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Create DataLoaders
def create_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, num_classes=34):
    try:
        train_dataset = SegmentationDataset(
            train_img_dir, 
            train_mask_dir, 
            num_classes=num_classes,
            transform=get_training_augmentation()
        )
        
        val_dataset = SegmentationDataset(
            val_img_dir, 
            val_mask_dir, 
            num_classes=num_classes,
            transform=get_validation_augmentation()
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,  # Reduced from 4 to improve stability
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,  # Reduced from 4 to improve stability
            pin_memory=True
        )
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        raise