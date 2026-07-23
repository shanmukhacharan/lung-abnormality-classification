"""
Dataset loading, preprocessing, and train/val/test splitting.

Preprocessing pipeline: resize to 224x224, convert grayscale -> 3-channel
RGB (required by the pretrained ResNet50 / ViT-B16 backbones), then a
70:15:15 train/val/test split.
"""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from . import config


def build_transform():
    """Return the preprocessing pipeline shared by all three models."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


def load_dataset(dataset_path: str = config.DATASET_PATH):
    """Load the full ImageFolder dataset with the shared transform applied."""
    transform = build_transform()
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    print("Total images:", len(dataset))
    print("Classes:", dataset.classes)
    return dataset


def split_dataset(dataset):
    """Stratified-by-size 70:15:15 train/val/test split."""
    dataset_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * dataset_size)
    val_size = int(config.VAL_SPLIT * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(dataset_path: str = config.DATASET_PATH):
    """Convenience wrapper: load, split, and wrap in DataLoaders."""
    dataset = load_dataset(dataset_path)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return dataset, train_loader, val_loader, test_loader
