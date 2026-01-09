# utils/data.py
# v12: Added augmentation support

import os
import zipfile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import glob

def get_transform(image_size=64, augment=False):
    """Transform with optional augmentation."""
    base = []
    
    if augment:
        base.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    
    base.extend([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(base)

def get_augment_transform(image_size=64):
    """Augmentation for consistency loss - stronger than training aug."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

class FolderDataset(Dataset):
    def __init__(self, root_path, transform=None, extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
        self.transform = transform
        self.images = []
        for ext in extensions:
            self.images.extend(glob.glob(os.path.join(root_path, '**', f'*{ext}'), recursive=True))
            self.images.extend(glob.glob(os.path.join(root_path, '**', f'*{ext.upper()}'), recursive=True))
        if not self.images:
            raise ValueError(f"No images found in {root_path}")
        print(f"Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

def load_dataset(dataset_name, data_path, zip_path=None, image_size=64, batch_size=128, augment=False):
    """Universal dataset loader."""
    transform = get_transform(image_size, augment=augment)
    
    if zip_path and os.path.exists(zip_path) and not os.path.exists(data_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(data_path))
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'auto':
        dataset_name = 'folder' if os.path.isdir(data_path) else ValueError("Cannot auto-detect")
    
    if dataset_name == 'celeba':
        if os.path.exists(os.path.join(data_path, 'img_align_celeba')):
            data_path = os.path.join(data_path, 'img_align_celeba')
        
        celeba_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5) if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if augment else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=os.path.dirname(data_path), transform=celeba_transform)
        
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'mnist':
        mnist_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=mnist_transform)
    elif dataset_name == 'folder':
        dataset = FolderDataset(data_path, transform=transform)
    elif dataset_name == 'imagefolder':
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True  # A100: increased workers
    )
    
    info = {'name': dataset_name, 'size': len(dataset), 'batches': len(loader), 'image_size': image_size}
    print(f"Loaded {info['name']}: {info['size']} samples, {info['batches']} batches")
    
    return loader, info

def load_from_config():
    from configs.config import DATASET_NAME, DATA_PATH, ZIP_PATH, IMAGE_SIZE, BATCH_SIZE, USE_AUGMENTATION
    return load_dataset(DATASET_NAME, DATA_PATH, ZIP_PATH, IMAGE_SIZE, BATCH_SIZE, augment=USE_AUGMENTATION)
