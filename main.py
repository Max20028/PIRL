import os
from dotenv import load_dotenv
from pathlib import Path

import data.data_processor as data_processor
import model.baseline as baseline

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



init_env_vars()
dataset = data_processor.process_data()
print("[SYSTEM] Starting training")
model = RLModel(dataset, batch_size=4)
model.train(epochs=5)
#baseline.train(dataset)
