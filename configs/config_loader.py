import yaml
from pathlib import Path

class Config:
    def __init__(self, config_dict):
        for k,v in config_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v) 


def load_config(path: str):
    path = Path(path)
    with open(path, 'r') as f: 
        config_dict= yaml.safe_load(f)

    return Config(config_dict)  