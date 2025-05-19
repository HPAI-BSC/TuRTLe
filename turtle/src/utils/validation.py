from typing import Dict, Any
from pydantic import ValidationError
from src.config.schemas import BenchmarkConfig, SlurmConfigurations

def validate_config(config: Dict[str, Any], schema: str) -> bool:
    """
    Validate the configuration against the specified schema.
    Parameters:
        - config (Dict[str, Any]): Configuration dictionary to validate.
        - schema (str): The schema to validate against ('benchmark' or 'slurm').
    Returns:
        - bool: True if validation is successful, False otherwise.
    """
    try:
        if schema == 'benchmark':
            BenchmarkConfig(**config)
        elif schema == 'slurm':
            SlurmConfigurations(**config)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False