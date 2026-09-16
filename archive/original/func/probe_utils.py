import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
import json
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import validate_repo_id

model_probe_config = {
    "llama3-8b": "llama3_1_8b_linear",
    "gemma2": "gemma2_9b_linear",
    "llama3-8b-instruct": "llama3_1_8b_linear",
    "gemma2-instruct": "gemma2_9b_linear",
}

def load_probe_head(
    probe_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda'
) -> Tuple[nn.Module, int]:
    """Load probe head from disk."""
    # Load probe config
    with open(probe_dir / "probe_config.json") as f:
        probe_config = json.load(f)
    
    hidden_size = probe_config['hidden_size']
    probe_layer_idx = probe_config['layer_idx']
    
    # Create probe head
    probe_head = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
    
    # Load weights
    state_dict = torch.load(
        probe_dir / "probe_head.bin",
        map_location="cpu",
        weights_only=True
    )
    probe_head.load_state_dict(state_dict)
    probe_head.eval()
    
    return probe_head, probe_layer_idx

# probe_dir = Path("/home/anwoy/HIDE/data/probes/llama3_1_8b_linear")
# probe_head, probe_layer_idx = load_probe_head(probe_dir)


LOCAL_PROBES_DIR = Path("/home/anwoy/HIDE/data/probes")
def download_probe_from_hf(
    repo_id: str,
    probe_id: Optional[str] = None,
    local_folder: Optional[Union[str, Path]] = None,
    hf_repo_subfolder_prefix: str = "",
    token: Optional[str] = None
) -> None:
    """Simplified probe download function for Modal."""
    api = HfApi()

    if local_folder is None:
        local_folder = LOCAL_PROBES_DIR / probe_id
    elif isinstance(local_folder, str):
        local_folder = Path(local_folder)

    local_folder.mkdir(parents=True, exist_ok=True)
    
    # List files in the repository subfolder
    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision="main"
    )
    print(f"Files in repo: {repo_files}")
    
    # Filter files by subfolder
    path_in_repo = os.path.join(hf_repo_subfolder_prefix, probe_id)
    subfolder_files = [f for f in repo_files if f.startswith(f"{path_in_repo}/")]
    print(f"Files in subfolder: {subfolder_files}")
    # Download each file
    for file_path in subfolder_files:
        print(file_path)
        # Get relative path within subfolder
        relative_path = file_path[len(path_in_repo):].lstrip('/')
        
        # Create subdirectory if needed
        local_file_path = local_folder / relative_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            token=token
        )
        
        # Copy to destination
        shutil.copy(downloaded_file, local_file_path)
    
    print(f"Downloaded probe to {local_folder}")

