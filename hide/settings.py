"""Paths are resolved on demand; imports never create directories."""
import os
from pathlib import Path


def data_folder():
    return str(Path(os.environ.get('HIDE_DATA_ROOT', 'data')).expanduser().resolve() / 'datasets')
