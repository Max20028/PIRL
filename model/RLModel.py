from torch.utils.data import DataLoader
from torch import nn
import torch
from model.UNet import UNet
from tqdm import tqdm

class RLModel:
    def __init__(self, dataset, batch_size=32):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get shapes from first sample
        self.T_in, self.H, self.W, self.C = dataset[0]["input_fields"].shape
        self.T_out = dataset[0]["output_fields"].shape[0]

        # Initialize model
        self.model = UNet(self.T_in, self.T_out, self.C).to(self.device)

        xs = []
        for i in range(0, 1000, 100):
            x = dataset[i]["input_fields"]
            xs.append(x)

        xs = torch.stack(xs)
        self.mu = xs.reshape(-1, self.C).mean(dim=0).to(self.device)
        self.std = xs.reshape(-1, self.C).std(dim=0).to(self.device)

        # Loss function - todo replace with -reward
        self.criterion = nn.MSELoss()

    def _preprocess(self, batch):
        return (batch - self.mu) / self.std

    def _postprocess(self, batch):
        return batch * self.std + self.mu

    def train(self, epochs=1, log_interval=10):
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(self.loader)
            pbar.set_description(f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                x = self._preprocess(batch["input_fields"].float().to(self.device))
                y_true = self._preprocess(batch["output_fields"].float().to(self.device))

                # Forward pass
                y_pred = self.model(x)

                # Compute loss
                loss = self.criterion(y_pred, y_true)

                # Backward pass
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.detach().item():.3f}")