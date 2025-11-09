import os
import sys
from typing import Dict, List, Optional


class TurtleCommandBuilder:
    """
    A class to dynamically build BigCode evaluation commands based on benchmark configuration.
    """

    def __init__(self, benchmark_config: Dict):
        """
        Initialize the command builder with the benchmark configuration.

        :param benchmark_config: A dictionary containing the benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    def get_launcher_command(self) -> List[str]:
        """
        Determine the launcher command based on the model type and execution mode.

        Parameters:
            kind_of_model: The type of model ('vllm' or 'llm').
            use_accelerate: Whether to use 'accelerate' for execution.

        return:
            The launcher command list of strings.
        """
        command_ = '"python3 -u turtle/src/turtle.py'

        return command_

    def build_dynamic_parameters(self, model_name: str) -> List[str]:
        """
        Build dynamic parameters for the command excluding slurm_config.

        Parameters:
            model_name: Name of the model to filter parameters for.
        Returns:
            A list of command-line arguments as strings.
        """
        params = []

        # 1. Validte the benchmark configuration
        if not hasattr(self, "benchmark_config") or not isinstance(self.benchmark_config, dict):
            raise ValueError("Configuration is not valid. Please check the benchmark configuration.")

        # 2. find the specific model configuration
        model_config = next(
            (m for m in self.benchmark_config.get("models", []) if m.get("name") == model_name),
            None,
        )

        if not model_config:
            raise ValueError(f"Model '{model_name}' was not found in the benchmark configuration.")

        # 3. Adding task name to de list
        task_name = self.benchmark_config.get("task")
        if task_name:
            params.append(f"--task {task_name}")

        # 4. List of excluded parameters
        excluded_params = {"name", "slurm_config", "multinode"}

        # 5. Add temperature parameter
        temperature = model_config.get("temperature")[0]

        # 6. Process each parameter in the model configuration
        for key, value in model_config.items():
            # check especial cases
            if key in excluded_params:
                continue

            # behavior for bolean values
            if isinstance(value, bool):
                if key == "save_metrics" and value:
                    # Add metric generations file
                    params.append(
                        f"--metric_output_path '{self.benchmark_config.get('metric_output_path')}{model_name}/metrics.json'"
                    )
                    continue
                elif key == "save_generations" and value:
                    # Add generations file
                    params.append(f"--{key} {value}")
                    params.append(
                        f"--save_generations_path '{self.benchmark_config.get('metric_output_path')}{model_name}/generations.json'"
                    )
                    continue
                elif key == "save_generations" and value == False:
                    params.append(f"--{key} False")
                    continue
                elif key == "save_references":
                    params.append(f"--{key} {value}")
                    continue
                else:
                    # Handle all other boolean parameters
                    params.append(f"--{key} {value}")
                    continue

            # behavior for list values
            if isinstance(value, list):
                if not value:  # Lista vacía
                    continue
                params.append(f"--{key} {value[0]}")
                continue

            # behavior for none values
            if value is None:
                continue

            # All rest of the parameters
            params.append(f"--{key} {value}")

        # 7. Add the dataset path and temp files
        path_data_benchmark = self.benchmark_config.get("path_data_benchmark")
        if path_data_benchmark:
            params.append(f"--path_data_benchmark {path_data_benchmark}")

        path_dataset_test = self.benchmark_config.get("path_dataset_test")
        if path_dataset_test:
            params.append(f"--path_dataset_test {path_dataset_test}")

        path_temporary_files = self.benchmark_config.get("path_temporary_files")
        if path_temporary_files:
            params.append(f"--path_temporary_files {path_temporary_files}")

        rtllm_version = self.benchmark_config.get("rtllm_version")
        if rtllm_version:
            params.append(f"--rtllm_version {rtllm_version}")

        generate_report = self.benchmark_config.get("generate_report")
        if generate_report:
            params.append(f"--generate_report {generate_report}")

        compute_ppl_only = self.benchmark_config.get("compute_ppl_only")
        if compute_ppl_only:
            params.append(f"--compute_ppl_only {compute_ppl_only}")

        simulator = self.benchmark_config.get("simulator")
        if simulator:
            params.append(f"--simulator {simulator}")

        return params

    def build_command(self, model: str) -> str:
        """
        Build the full BigCode command based on the provided configuration.

        Parameters:
            task_name: Name of the specific task to run (optional).
            use_accelerate: Boolean indicating whether to use accelerate or plain Python.
        return:
            The complete command list of string.
        """
        # Get the execution command prefix
        execution_command = self.get_launcher_command()

        # Build dynamic parameters
        dynamic_params = self.build_dynamic_parameters(model)

        # Add global parameters (e.g., model path)
        global_params = [
            f"--model {self.benchmark_config['path_model']}{model}",
        ]

        # Combine all parts into the final command
        full_command = execution_command + " " + " ".join(global_params) + " " + " ".join(dynamic_params)
        return full_command + '"'
