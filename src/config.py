"""
Shared configuration for the lung abnormality classification project.

Update DATASET_PATH to point at your local copy of the dataset before
running any of the scripts in this folder.
"""

import torch

# Path to the dataset root, expected to contain one sub-folder per class
# (COVID, Lung_Opacity, Normal, Viral Pneumonia).
DATASET_PATH = "./data/Chest X-Ray Image"

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
# TEST_SPLIT is whatever remains (0.15)

LEARNING_RATE = 1e-4
EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "./best_cnn_model.pth"
