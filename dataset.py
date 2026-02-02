"""
Dataset pour charger les images médicales et leurs masques
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MedicalDataset(Dataset):
    """Dataset pour images médicales"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Charger image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('L')
        
        # Charger mask
        mask_name = img_name.replace('.tif', '_mask.tif').replace('.png', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Binariser le mask
        mask = (mask > 0.5).float()
        
        return image, mask


def get_transforms():
    """Transformations pour les images"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


def get_dataloaders(batch_size=4):
    """Crée les dataloaders"""
    
    transform = get_transforms()
    
    train_dataset = MedicalDataset(
        'data/train/images',
        'data/train/masks',
        transform=transform
    )
    
    val_dataset = MedicalDataset(
        'data/val/images',
        'data/val/masks',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
