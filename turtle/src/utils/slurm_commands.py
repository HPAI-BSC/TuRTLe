from typing import Dict, List, Optional

class SlurmConfigLoader:
    """
    A class to load SLURM configurations from a file and generate sbatch commands.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the SLURM configuration loader with the provided configuration.

        Args:
            config: A dictionary containing the SLURM configurations.
        """
        self.config = config.get("configurations", [])
        if not self.config:
            raise ValueError("No SLURM configurations found in the provided file.")
        
        # Preprocess configurations into a dictionary
        self.config_dict = self._create_config_dict()

    def _get_slurm_parameters(self, slurm_config: Dict) -> List[str]:
        """
        Extract SLURM parameters from the configuration and format them for the sbatch command.

        Args: 
            slurm_config: A dictionary representing a specific SLURM configuration.
        Returns: 
            A list of SLURM parameters as strings.
        """
        params = []
        for key, value in slurm_config.items():
            if key == "type" or key == "singularity_path":  # Exclude these from SLURM parameters
                continue
            if isinstance(value, bool):  # Handle boolean flags
                if value:
                    params.append(f"--{key} ")
            elif value is not None:  # Handle other parameters
                params.append(f"--{key}={value}")
        return params

    def _create_config_dict(self) -> Dict[str, str]:
        """
        Create a dictionary mapping configuration types to their sbatch commands.

        Returns:
            Dictionary where keys are configuration types and values are sbatch command strings.
        """
        config_dict = {}
        for cfg in self.config:
            if "type" not in cfg:
                continue
            config_type = cfg["type"]
            slurm_params = self._get_slurm_parameters(cfg)
            config_dict[config_type] = f"sbatch {' '.join(slurm_params)}"
        return config_dict

    def get_slurm_command(self, config_type: str, wrap_command: Optional[str] = None) -> str:
        """
        Get the sbatch command for the specified configuration type.

        Args:
            config_type: The type of configuration (e.g., 'general-single-node-small').
            wrap_command: Command to wrap in the sbatch execution (optional).
        
        Returns:
            The complete sbatch command as a string.
        
        Raises:
            ValueError: If the configuration type is not found.
        """
        if config_type not in self.config_dict:
            raise ValueError(f"No SLURM configuration found for type: {config_type}")
        
        command = self.config_dict[config_type]
        if wrap_command:
            command += f' --wrap="{wrap_command}"'
        
        return command

    def get_all_configs(self) -> Dict[str, str]:
        """
        Get all configurations as a dictionary mapping types to sbatch commands.
        
        Returns:
            Dictionary of all configuration types and their commands.
        """
        return self.config_dict.copy()