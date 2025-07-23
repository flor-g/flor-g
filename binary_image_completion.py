import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Parameters
IMAGE_SIZE = 100  # 100x100 images
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_SAMPLES = 80
HIDDEN_UNITS = 512
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MASK_PIXELS = 80  # Number of white pixels revealed in each input
SEGMENT_WIDTH = np.pi  # Width of sine segment for each sample


def generate_dataset(num_samples: int):
    """Generate sine-wave datasets composed of shifted segments.

    For each sample a sine curve with a constant amplitude is drawn over a
    window of width ``SEGMENT_WIDTH``. The start of this window is chosen at
    random so that each sample contains a different portion of ``sin(x)``.
    ``MASK_PIXELS`` white pixels from the generated curve image are then
    revealed while the rest are set to ``0`` in the input. The full segment
    image is used as the training target.
    """

    inputs = np.zeros((num_samples, NUM_PIXELS), dtype=np.float32)
    targets = np.zeros((num_samples, NUM_PIXELS), dtype=np.float32)

    amplitude = IMAGE_SIZE * 0.4
    center = IMAGE_SIZE / 2.0

    for i in range(num_samples):
        start = np.random.uniform(0.0, 2 * np.pi - SEGMENT_WIDTH)
        x_vals = np.linspace(start, start + SEGMENT_WIDTH, IMAGE_SIZE)

        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for x_pixel, x in enumerate(x_vals):
            y = amplitude * np.sin(x)
            y_pos = center - y
            lower = int(np.floor(y_pos))
            upper = int(np.ceil(y_pos))
            if 0 <= lower < IMAGE_SIZE:
                image[lower, x_pixel] = 1.0
            if 0 <= upper < IMAGE_SIZE:
                image[upper, x_pixel] = 1.0

        flat_image = image.reshape(NUM_PIXELS)
        targets[i] = flat_image

        mask = np.zeros(NUM_PIXELS, dtype=np.float32)
        white_indices = np.flatnonzero(flat_image)
        if len(white_indices) >= MASK_PIXELS:
            reveal_indices = np.random.choice(white_indices, MASK_PIXELS, replace=False)
        else:
            reveal_indices = white_indices
        mask[reveal_indices] = 1.0
        inputs[i] = flat_image * mask

    return inputs, targets


class TwoLayerNet(nn.Module):
    """Simple two-layer fully connected neural network."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_PIXELS, HIDDEN_UNITS)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_UNITS, NUM_PIXELS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_inputs.size(0)
    return total_loss / len(dataloader.dataset)


def visualize_results(model, input_image, target_image):
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(input_image).unsqueeze(0)).squeeze(0).numpy()
    output_image = (output > 0.5).astype(np.float32)  # Threshold for binary output

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(target_image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(input_image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    axes[1].set_title('Input (Masked)')
    axes[1].axis('off')

    axes[2].imshow(output_image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    axes[2].set_title('Predicted')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate sine-wave dataset.
    inputs, targets = generate_dataset(NUM_SAMPLES)

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs)
    targets_tensor = torch.from_numpy(targets)

    # Split into training and validation sets
    train_size = int(0.8 * NUM_SAMPLES)
    val_size = NUM_SAMPLES - train_size
    train_dataset = TensorDataset(inputs_tensor[:train_size], targets_tensor[:train_size])
    val_dataset = TensorDataset(inputs_tensor[train_size:], targets_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, loss, optimizer
    model = TwoLayerNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Visualize results on a random validation sample
    idx = np.random.randint(0, val_size)
    input_img = inputs[train_size + idx]
    target_img = targets[train_size + idx]
    visualize_results(model, input_img, target_img)
