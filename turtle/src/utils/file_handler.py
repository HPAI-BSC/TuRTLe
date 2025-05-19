from pathlib import Path
import shutil
import yaml
import logging
from typing import Dict, Any

class FileHandler:
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        if not file_path.exists():
            # loggin error
            logging.error(f"File '{file_path}' does not exist.")
            return {}
        else:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)