import yaml
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")

def load_config(path=CONFIG_PATH):
    """Loads configuration from YAML file and includes BASE_DIR."""
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    config["BASE_DIR"] = BASE_DIR
    return config
