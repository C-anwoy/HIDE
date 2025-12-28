import getpass
import os
import sys

# __USERNAME = getpass.getuser()
_BASE_DIR = f'/home/anwoy/HIDE/data'
MODEL_PATH = f'/home/models/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
ABLATION_FOLDER = os.path.join(GENERATION_FOLDER, 'ablations')
os.makedirs(ABLATION_FOLDER, exist_ok=True)
PROBE_FOLDER = os.path.join(_BASE_DIR, 'probes')
os.makedirs(PROBE_FOLDER, exist_ok=True)