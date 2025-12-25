# utils/data.py
# Universal data loader - works with any image dataset

import os
import zipfile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import glob

def get_transform(image_size=64):
    """Standard transform for any image dataset."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

class FolderDataset(Dataset):
    """Load images from any folder structure."""
    def __init__(self, root_path, transform=None, extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
        self.transform = transform
        self.images = []
        
        # Recursively find all images
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
        return img, 0  # Return dummy label for compatibility

def load_dataset(dataset_name, data_path, zip_path=None, image_size=64, batch_size=128):
    """
    Universal dataset loader.
    
    Args:
        dataset_name: 'celeba', 'cifar10', 'mnist', 'folder', or 'auto'
        data_path: Path to data
        zip_path: Optional zip file to extract first
        image_size: Target image size
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader, dataset_info dict
    """
    transform = get_transform(image_size)
    
    # Extract zip if provided and path doesn't exist
    if zip_path and os.path.exists(zip_path) and not os.path.exists(data_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(data_path))
    
    dataset_name = dataset_name.lower()
    
    # Auto-detect
    if dataset_name == 'auto':
        if os.path.isdir(data_path):
            dataset_name = 'folder'
        else:
            raise ValueError(f"Cannot auto-detect dataset type for {data_path}")
    
    # Load based on type
    if dataset_name == 'celeba':
        # CelebA uses ImageFolder structure
        if os.path.exists(os.path.join(data_path, 'img_align_celeba')):
            data_path = os.path.join(data_path, 'img_align_celeba')
        dataset = datasets.ImageFolder(root=os.path.dirname(data_path), transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]))
        
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        
    elif dataset_name == 'mnist':
        # Convert grayscale to RGB
        mnist_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale -> RGB
        ])
        dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=mnist_transform)
        
    elif dataset_name == 'fashion_mnist':
        mnist_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=mnist_transform)
        
    elif dataset_name == 'folder':
        # Generic folder of images
        dataset = FolderDataset(data_path, transform=transform)
        
    elif dataset_name == 'imagefolder':
        # PyTorch ImageFolder format (class subfolders)
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'celeba', 'cifar10', 'mnist', 'folder', or 'auto'")
    
    # Create loader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True,  # Avoid small batches that break BOM
    )
    
    info = {
        'name': dataset_name,
        'size': len(dataset),
        'batches': len(loader),
        'image_size': image_size,
    }
    
    print(f"Loaded {info['name']}: {info['size']} samples, {info['batches']} batches")
    
    return loader, info


def load_from_config():
    """Load dataset using config.py settings."""
    from configs.config import DATASET_NAME, DATA_PATH, ZIP_PATH, IMAGE_SIZE, BATCH_SIZE
    return load_dataset(DATASET_NAME, DATA_PATH, ZIP_PATH, IMAGE_SIZE, BATCH_SIZE)
