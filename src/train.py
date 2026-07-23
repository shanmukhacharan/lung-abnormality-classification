"""
Shared training loop used for all three models: Adam optimiser,
cross-entropy loss, and a fixed number of epochs (see src/config.py).
"""

import torch.nn as nn
import torch.optim as optim

from . import config


def train_model(model, train_loader, epochs: int = config.EPOCHS, lr: float = config.LEARNING_RATE,
                 trainable_params=None):
    """
    Train `model` on `train_loader` for `epochs` epochs.

    trainable_params: pass a specific parameter subset to optimise (e.g.
    model.fc.parameters() for the frozen baseline CNN). Defaults to all
    of the model's parameters.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params or model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    return model
