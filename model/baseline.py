    
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from neuralop.models import FNO
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download
import os
from pathlib import Path

def train(dataset):
    xs = []
    F = dataset.metadata.n_fields
    device = "cuda"
    base_path = (Path(__file__).parent / "../data/raw/").resolve()
    print(base_path)

    for i in range(0, 1000, 100):
        x = dataset[i]["input_fields"]
        xs.append(x)

    xs = torch.stack(xs)
    mu = xs.reshape(-1, F).mean(dim=0).to(device)
    sigma = xs.reshape(-1, F).std(dim=0).to(device)

    def preprocess(x):
        return (x - mu) / sigma

    def postprocess(x):
        return sigma * x + mu

    model = FNO(
        n_modes=(16, 16),
        in_channels=4 * F,
        out_channels=1 * F,
        hidden_channels=128,
        n_layers=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4,
        num_workers=4,
    )

    for epoch in range(1):
        for batch in (bar := tqdm(train_loader)):
            x = batch["input_fields"]
            x = x.to(device)
            x = preprocess(x)
            x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

            y = batch["output_fields"]
            y = y.to(device)
            y = preprocess(y)
            y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

            fx = model(x)

            mse = (fx - y).square().mean()
            mse.backward()

            optimizer.step()
            optimizer.zero_grad()

            bar.set_postfix(loss=mse.detach().item())
        device = "cuda"

    base_path = os.path.join(base_path, "turbulent_radiative_layer_2D/datasets")
    print(base_path)
    validset = WellDataset(
        well_base_path=base_path,
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=False,
    )

    item = validset[123]

    x = item["input_fields"]
    x = x.to(device)
    x = preprocess(x)
    x = rearrange(x, "Ti Lx Ly F -> 1 (Ti F) Lx Ly")

    y = item["output_fields"]
    y = y.to(device)

    with torch.no_grad():
        fx = model(x)
        fx = rearrange(fx, "1 (To F) Lx Ly -> To Lx Ly F", F=F)
        fx = postprocess(fx)

    VRMSE.eval(fx, y, meta=validset.metadata)