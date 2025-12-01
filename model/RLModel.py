from torch.utils.data import IterableDataset, DataLoader
from torch import nn
import torch
from model.UNet import UNet
from tqdm import tqdm

import random

from reward_func import compare

class CombinedBatchDataset(IterableDataset):
    def __init__(self, loaders):
        super().__init__()
        self.loaders = loaders
        self.batch_counts = [len(loader) for loader in loaders]
        self.total_batches = sum(len(loader) for loader in loaders)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        # Generate random order for loader selection
        loader_order = []
        for idx, count in enumerate(self.batch_counts):
            loader_order += [idx] * count
        random.shuffle(loader_order)

        # Create iterators
        iters = [iter(loader) for loader in self.loaders]

        for loader_idx in loader_order:
            yield next(iters[loader_idx])

class RLModel:


    def __init__(self, datasets, num_channels=3, batch_size=4):
        loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
        self.dataset = CombinedBatchDataset(loaders)
        self.loader = DataLoader(self.dataset, batch_size=None, num_workers=0, pin_memory=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get shapes from first sample
        self.T_in, self.H, self.W, _ = datasets[0][0]["input_fields"].shape
        self.T_out = datasets[0][0]["output_fields"].shape[0]

        self.C = num_channels
        # Hardcode to three to only keep pressure, velocity_x, velocity_y

        # Initialize model
        self.model = UNet(self.T_in, self.T_out, self.C).to(self.device)
        # we are hard-coding it to 3 channels, pressure, velocity_x, velocity_y

        # Loss function - todo replace with -reward
        self.criterion = nn.MSELoss()

    def train(self, epochs=1, log_interval=10):
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(self.loader, total=len(self.loader))
            pbar.set_description(f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                x = batch["input_fields"].float().to(self.device, non_blocking=True)
                y_true = batch["output_fields"].float().to(self.device, non_blocking=True)
                x = x[..., -self.C:]
                y_true = y_true[..., -self.C:]

                # Forward pass
                y_pred = self.model(x)

                # Compute loss
                #loss = 0.5 * compare(y_true[0, 0, ..., 1:], y_pred[0, 0, ..., 1:]).mean() + 0.5 * torch.mean((y_true[0, 0, ..., 0] - y_pred[0, 0, ..., 0])**2)
                # Currently equally weighting pressure loss and velocity loss

                loss = self.criterion(y_pred, y_true)

                # Backward pass
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.detach().item():.3f}")
        torch.save(self.model.state_dict(), "model.pth")