import os
from dotenv import load_dotenv
from pathlib import Path

#import data.data_processor as data_processor
import model.baseline as baseline
import random


from the_well.data import WellDataset
from torch.utils.data import Subset, ConcatDataset, DataLoader

from model.RLModel import RLModel

def init_env_vars():
    if 'INITIALIZED' not in os.environ:
        os.environ['INITIALIZED'] = '1'
        data_raw_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/raw"))
        datasets_initialized = '0'
        
        if data_raw_path.is_dir():
            if any(item.is_dir() for item in data_raw_path.iterdir()):
                datasets_initialized = '1'

        os.environ['DATASETS_INITIALIZED'] = datasets_initialized



# init_env_vars()
# ds1 = data_processor.process_data("turbulent_radiative_layer_2D")
# ds2 = data_processor.process_data("shear_flow")
# ds3 = data_processor.process_data("rayleigh_benard")
# ds4 = data_processor.process_data("acoustic_scattering_discontinuous")


# test_set = data_processor.process_data("acoustic_scattering_discontinuous", split="test")

if __name__ == '__main__':
    subsets = []
    for name in ["turbulent_radiative_layer_2D", "shear_flow", "rayleigh_benard"]:
        print(f"Fetching Dataset {name}")
        ds = WellDataset(
                well_base_path="hf://datasets/polymathic-ai/",
                well_dataset_name=name,
                well_split_name="train",
                use_normalization=False,
                n_steps_input=4,
                n_steps_output=1,
                # other dataset kwargs as needed
            )
        idxs = random.sample(range(len(ds)), 500)
        subsets.append(Subset(ds, idxs))
        print(f"Done fetching dataset {name}")

    # Last two fields are pressure and velocity so when i am loading a batch i will take only the last two

    print("[SYSTEM] Starting training")
    model = RLModel(subsets)
    model.train(epochs=1)
    #baseline.train(dataset)
