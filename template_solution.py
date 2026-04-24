"""
CITATION DISCLAIMER: AI USAGE

The implementation of this script was supported by various AI tools, including
ChatGPT, Gemini, and Claude. The underlying logic and problem-solving approach
were developed by the students.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import copy
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(**kwargs):
  
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    train_data = train_data / 255.0

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    # Normalize to [0, 1]
    test_data_input = test_data_input / 255.0

    train_data_label = train_data.clone()
    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0.0

    # Add a mask channel: 1 inside the hole, 0 elsewhere. Lets the model
    # distinguish "real black pixel" from "missing pixel".
    def _with_mask(img):
        mask = torch.zeros_like(img)
        mask[:, :, 10:18, 10:18] = 1.0
        return torch.cat([img, mask], dim=1)

    train_data_input = _with_mask(train_data_input)
    test_data_input = _with_mask(test_data_input)

    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i, 0], cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


def training(train_data_input, train_data_label, **kwargs):

    model = Model()
    model.train()
    model.to(device)


    def criterion(output, target):
        return F.mse_loss(output[:, :, 10:18, 10:18],target[:, :, 10:18, 10:18])

    # DONE: Using a Adam optimizer for now (momentum, adaptive learning rate SGD)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    dataset = TensorDataset(train_data_input, train_data_label)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    n_epochs = 20

    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        epoch_loss_sum = 0.0
        n_batches = 0
        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        mean_loss = epoch_loss_sum / n_batches
        marker = ""
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
            marker = " (best)"
        print(f"Epoch {epoch} mean loss: {mean_loss:.6f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

class Model(nn.Module):
    """
    Small conv encoder-decoder for 28x28 inpainting.
    """

    def __init__(self):
        super().__init__()

        # Encoder: 28x28 -> 14x14 (input is 2 channels: image + hole mask)
        self.enc1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Decoder: 14x14 -> 28x28
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = self.dec2(x)
        return x


def testing(model, test_data_input):
   
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []

        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Drop the mask channel now that inference is done; downstream code and
    # the shape assertion expect a 1-channel input.
    test_data_input = test_data_input[:, :1]

    # Change outer ring back to what we were given as input
    center = test_data_output[:, :, 10:18, 10:18].clamp(0, 1)
    test_data_output = test_data_input.clone()
    test_data_output[:, :, 10:18, 10:18] = center

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Since data was normalized to [0, 1], scale it back to [0, 255] before saving
    test_data_output *= 255.0
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i, 0].cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = load_data()
    # Train the model
    model = training(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    testing(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
