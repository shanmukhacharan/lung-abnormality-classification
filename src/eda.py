"""
Exploratory data analysis: class distribution, sample images, pixel
intensity distribution, average brightness per class, and Canny edge
detection.
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import config


def class_counts(dataset_path: str = config.DATASET_PATH) -> dict:
    """Count images per class folder."""
    counts = {}
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            counts[folder] = len(os.listdir(folder_path))
    return counts


def plot_class_distribution(counts: dict):
    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, values)
    plt.title("Chest X-ray Dataset Class Distribution")
    plt.xlabel("Disease Category")
    plt.ylabel("Number of Images")
    plt.show()


def plot_sample_images(dataset_path: str = config.DATASET_PATH):
    classes = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        folder = os.path.join(dataset_path, cls)
        image_name = random.choice(os.listdir(folder))
        image = cv2.imread(os.path.join(folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(cls)
        plt.axis("off")

    plt.suptitle("Sample Chest X-ray Images from Each Class")
    plt.show()


def pixel_intensity_distribution(dataset_path: str = config.DATASET_PATH, sample_size: int = 200):
    sample_images = []
    for cls in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, cls)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            sample_images.append(os.path.join(class_folder, img_name))

    sample_images = random.sample(sample_images, sample_size)

    pixel_values = []
    for img_path in sample_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pixel_values.extend(img.flatten())

    pixel_values = np.array(pixel_values)
    print("Pixels analysed:", len(pixel_values))

    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=50)
    plt.title("Pixel Intensity Distribution in Chest X-ray Dataset")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    return pixel_values


def average_brightness_per_class(dataset_path: str = config.DATASET_PATH, classes=None, sample_size: int = 150):
    classes = classes or config.CLASSES
    brightness = []

    for cls in classes:
        class_folder = os.path.join(dataset_path, cls)
        values = []
        for img_name in os.listdir(class_folder)[:sample_size]:
            img = cv2.imread(os.path.join(class_folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                values.append(np.mean(img))
        brightness.append(np.mean(values))

    plt.figure(figsize=(8, 5))
    plt.bar(classes, brightness)
    plt.title("Average Image Brightness per Disease Class")
    plt.xlabel("Disease Category")
    plt.ylabel("Average Pixel Intensity")
    plt.show()

    return brightness


def plot_edge_detection(dataset_path: str = config.DATASET_PATH, classes=None):
    classes = classes or config.CLASSES

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        class_folder = os.path.join(dataset_path, cls)
        img_name = random.choice(os.listdir(class_folder))
        img = cv2.imread(os.path.join(class_folder, img_name), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 50, 150)

        plt.subplot(2, 2, i + 1)
        plt.imshow(edges, cmap="gray")
        plt.title(f"Edges - {cls}")
        plt.axis("off")

    plt.suptitle("Edge Detection of Lung Structures")
    plt.show()
