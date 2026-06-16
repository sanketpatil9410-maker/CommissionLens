import yaml
import os

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

def ensure_directories_exist(directories: list):
    """Ensure that a list of directories exists."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
