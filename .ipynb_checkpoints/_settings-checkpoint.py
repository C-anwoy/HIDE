import getpass
import os
import sys

# __USERNAME = getpass.getuser()
_BASE_DIR = f'./data'
MODEL_PATH = f''
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
ABLATION_FOLDER = os.path.join(GENERATION_FOLDER, 'ablations')
os.makedirs(ABLATION_FOLDER, exist_ok=True)