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
HIDDEN_UNITS2 = 256
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MASK_PIXELS = 20  # Number of white pixels revealed in each input
SEGMENT_WIDTH = 2*(np.pi)  # Width of sine segment for each sample


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

    amplitude = IMAGE_SIZE * (0.3 + 0.2 * np.random.rand())
    center = IMAGE_SIZE / 2.0

    for i in range(num_samples):
        start = np.random.uniform(0.0, 4 * np.pi - SEGMENT_WIDTH)
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


def apply_random_mask(images: np.ndarray) -> np.ndarray:
    """Create masked inputs from target images using random white pixel selection."""

    masked = np.zeros_like(images)
    for i, flat_image in enumerate(images):
        mask = np.zeros(NUM_PIXELS, dtype=np.float32)
        white_indices = np.flatnonzero(flat_image)
        if len(white_indices) >= MASK_PIXELS:
            reveal_indices = np.random.choice(white_indices, MASK_PIXELS, replace=False)
        else:
            reveal_indices = white_indices
        mask[reveal_indices] = 1.0
        masked[i] = flat_image * mask
    return masked


class TwoLayerNet(nn.Module):
    """Convolutional autoencoder for binary image completion."""

    def __init__(self):
        super().__init__()
        # Encoder downsamples the 100x100 input to 32 feature maps of size 25x25
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )

        # Flatten encoder output and pass through dense bottleneck
        self.enc_flat_features = 32 * 25 * 25
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Sequential(
            nn.Linear(self.enc_flat_features, 256),
            nn.Tanh(),
            nn.Linear(256, self.enc_flat_features),
            nn.Tanh(),
        )

        # Decoder upsamples back to the original spatial resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input comes in flattened; reshape for convolutional layers
        x = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

        # Encode to feature map representation
        enc_out = self.encoder(x)

        # Bottleneck fully connected layers
        flat = self.flatten(enc_out)
        flat = self.fc_enc(flat)

        # Expand back to feature map size
        flat = flat.view(-1, 32, 25, 25)

        # Skip connection from encoder output to decoder input
        dec_in = flat + enc_out

        # Decode to reconstructed image
        x = self.decoder(dec_in)
        x = self.sigmoid(x)

        # Return flattened output for loss calculation
        return x.view(-1, NUM_PIXELS)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Dice Loss for binary predictions.

    Args:
        pred: Tensor containing model predictions with values in ``[0, 1]``.
        target: Tensor of ground truth labels with values ``0`` or ``1``.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice loss averaged over the batch.
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score.mean()


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
    _, targets = generate_dataset(NUM_SAMPLES)

    # Split targets into training and validation sets
    train_size = int(0.8 * NUM_SAMPLES)
    val_size = NUM_SAMPLES - train_size
    train_targets = targets[:train_size]
    val_targets = targets[train_size:]

    # Initialize model, loss, optimizer
    model = TwoLayerNet()
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_weights = None
    best_weights_path = "best_model.pth"

    # Training loop
    for epoch in range(EPOCHS):
        # Randomly mask inputs each epoch
        train_inputs = torch.from_numpy(apply_random_mask(train_targets))
        val_inputs = torch.from_numpy(apply_random_mask(val_targets))

        train_dataset = TensorDataset(train_inputs, torch.from_numpy(train_targets))
        val_dataset = TensorDataset(val_inputs, torch.from_numpy(val_targets))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_weights, best_weights_path)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

    if best_weights is not None:
        model.load_state_dict(torch.load(best_weights_path))

    # Visualize results on a random validation sample
    idx = np.random.randint(0, val_size)
    masked = apply_random_mask(val_targets[idx : idx + 1])[0]
    input_img = masked
    target_img = val_targets[idx]
    visualize_results(model, input_img, target_img)
