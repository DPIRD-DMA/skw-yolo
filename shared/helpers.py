"""Visualization and system info helpers."""

import platform

import fastai
import torch
from fastai.torch_core import default_device


def print_system_info():
    """Print system and library information."""
    info = {
        "PyTorch Version": torch.__version__,
        "CUDA Available": "Yes" if torch.cuda.is_available() else "No",
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "Python Version": platform.python_version(),
        "Fastai Version": fastai.__version__,
        "Default Device": str(default_device()),
        "Device Name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "N/A",
    }

    max_key_length = max(len(key) for key in info.keys())
    print("System Information")
    print("-" * (max_key_length + 20))
    for key, value in info.items():
        print(f"{key.ljust(max_key_length)} : {value}")
    print("-" * (max_key_length + 20))
