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
MASK_FRACTION = 0.3  # Fraction of pixels to mask in input


def generate_dataset(num_samples: int):
    """Generate sine-wave binary images and corresponding masked inputs.

    Each sample contains a single sine curve drawn on a blank image. The curve
    is defined by randomly chosen vertical amplitude and horizontal scaling
    factors. Pixels on or nearest to the curve are set to ``1``. A fraction of
    pixels specified by ``MASK_FRACTION`` are then removed from the image to
    produce the masked input.
    """

    # Container for images before flattening
    images = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    x_vals = np.linspace(0.0, np.pi, IMAGE_SIZE)

    for i in range(num_samples):
        amplitude = np.random.uniform(IMAGE_SIZE * 0.1, IMAGE_SIZE * 0.4)
        scale = np.random.uniform(0.5, 2.0)
        center = IMAGE_SIZE / 2.0
        for x_pixel, x in enumerate(x_vals):
            y = amplitude * np.sin(scale * x)
            y_pos = center - y
            lower = int(np.floor(y_pos))
            upper = int(np.ceil(y_pos))
            if 0 <= lower < IMAGE_SIZE:
                images[i, lower, x_pixel] = 1.0
            if 0 <= upper < IMAGE_SIZE:
                images[i, upper, x_pixel] = 1.0

    # Flatten images for the network
    flat_images = images.reshape(num_samples, NUM_PIXELS)

    # Create masked versions for inputs
    masks = np.random.rand(num_samples, NUM_PIXELS) > MASK_FRACTION
    inputs = flat_images * masks.astype(np.float32)
    return inputs, flat_images


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
