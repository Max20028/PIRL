# !pip install the_well[benchmark]

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

import os
import sys


def process_data():
    device = "cuda"
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw/")

    if os.environ.get('INITIALIZED', '0') == '0':
        print("[ERROR] Environment variables not set")
        sys.exit()
    elif os.environ.get('DATASETS_INITIALIZED', '0') == '0':
        print("[SYSTEM] Downloading datasets")
        well_download(base_path=os.path.join(base_path, "turbulent_radiative_layer_2D"), dataset="turbulent_radiative_layer_2D", split="train")
        #well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="valid")
        #well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="test")
        os.environ['DATASETS_INITIALIZED'] = '1'
        print("[SYSTEM] Finished downloading datasets")
    else:
        print("[SYSTEM] Datasets already initialized")


    # fyi WellDataset is instanceo f torch.utils.data.Dataset
    # item = dataset[0]
    # list(item.keys()) ->
    # ['input_fields',
    #  'output_fields',
    #  'constant_scalars',
    #  'boundary_conditions',
    #  'space_grid',
    #  'input_time_grid',
    #  'output_time_grid']
    print(os.path.join(base_path, "turbulent_radiative_layer_2D/datsets"))
    dataset = WellDataset(
        well_base_path=os.path.join(base_path, "turbulent_radiative_layer_2D/datasets"),
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=False,
    )

    field_names = [name for group in dataset.metadata.field_names.values() for name in group]

    F = dataset.metadata.n_fields
    x = dataset[42]["input_fields"]
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")

    fig, axs = plt.subplots(F, 4, figsize=(4 * 2.4, F * 1.2))

    for field in range(F):
        vmin = np.nanmin(x[field])
        vmax = np.nanmax(x[field])

        axs[field, 0].set_ylabel(f"{field_names[field]}")

        for t in range(4):
            axs[field, t].imshow(
                x[field, t], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax
            )
            axs[field, t].set_xticks([])
            axs[field, t].set_yticks([])

            axs[0, t].set_title(f"xtx_{t}xt​")

    plt.tight_layout()
    plt.show()
    return dataset




