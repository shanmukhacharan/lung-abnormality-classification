"""
Grad-CAM implementation for the fine-tuned ResNet50 CNN. Hooks the last
convolutional block (layer4) to generate class-activation heatmaps that
show which lung regions drove each prediction.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_gradcam(model, image, target_layer=None):
    """
    Run Grad-CAM for a single image tensor (shape [1, C, H, W]).

    Returns (cam, predicted_class) where cam is a 224x224 normalised
    heatmap array.
    """
    gradients, activations = [], []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = target_layer or model.layer4[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(image)
    pred_class = torch.argmax(output)
    score = output[0, pred_class]

    model.zero_grad()
    score.backward()

    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    return cam, pred_class.item()


def plot_gradcam_single(model, image, true_label=None, device=None):
    """Plot original / heatmap / overlay for a single sample (Figure 11 in the report)."""
    model.eval()
    img_tensor = image.to(device) if device else image

    cam, pred_class = generate_gradcam(model, img_tensor)

    img = img_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255
    overlay = heatmap * 0.4 + img

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.show()

    if true_label is not None:
        print("True label:", true_label)
    print("Predicted label:", pred_class)


def plot_gradcam_all_classes(model, test_loader, class_names, device):
    """Plot one Grad-CAM overlay per class (Figure 12 in the report)."""
    model.eval()
    samples = {}

    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            if lbl.item() not in samples:
                samples[lbl.item()] = img.unsqueeze(0)
            if len(samples) == len(class_names):
                break
        if len(samples) == len(class_names):
            break

    plt.figure(figsize=(12, 8))

    for i, (label, img) in enumerate(samples.items()):
        img = img.to(device)
        cam, pred = generate_gradcam(model, img)

        original = img.cpu().numpy()[0].transpose(1, 2, 0)
        original = (original - original.min()) / (original.max() - original.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = heatmap / 255
        overlay = heatmap * 0.4 + original

        plt.subplot(2, 2, i + 1)
        plt.imshow(overlay)
        plt.title(f"True: {class_names[label]} | Pred: {class_names[pred]}")
        plt.axis("off")

    plt.suptitle("Grad-CAM Explanations for All Classes", fontsize=16)
    plt.show()
