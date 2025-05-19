import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from src.utils.file_handler import FileHandler
from src.utils.registry_updater import RegistryUpdater
from src.utils.slurm_commands import SlurmConfigLoader
from src.utils.turtle_commands import TurtleCommandBuilder


class WorkflowExecutor:
    """This class is responsible for executing python workflows
    using subprocess."""

    def __init__(self, args: Dict, logger: logging.Logger) -> None:
        self.args = args
        self.file_handler = FileHandler()
        self.registry_updater = RegistryUpdater()
        self.CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
        self.logger = logger

    def load_yaml_config(self, file_path: Path) -> Dict:
        """Load a YAML configuration file.
        Parameters:
            - file_path: Path to the YAML file.
        Returns:
            - config: A dictionary containing the loaded configuration.
        """
        return self.file_handler.load_yaml(file_path)

    def build_singularity_commands(
        self, task_image_list: list[tuple[str, str]]
    ) -> dict[str, str]:
        """Build a dictionary of Singularity commands for each task.

        Args:
            task_image_list: List of tuples where each tuple contains (task_name, image_path)

        Returns:
            Dictionary mapping task names to their corresponding Singularity commands
        """
        commands = {}
        for task_name, image_path in task_image_list:
            commands[task_name] = (
                f"VLLM_WORKER_MULTIPROC_METHOD=spawn singularity exec --nv {image_path} bash -c "
            )
        return commands

    def build_docker_commands(
        self, task_image_list: list[tuple[str, str]]
    ) -> dict[str, str]:
        """Build a dictionary of Docker commands for each task.

        Args:
            task_image_list: List of tuples where each tuple contains (task_name, image_name)

        Returns:
            Dictionary mapping task names to their corresponding Docker commands
        """
        commands = {}
        for task_name, image_name in task_image_list:
            commands[task_name] = f"docker run --gpus all {image_name} bash -c "
        return commands

    def run_command(self, command: str) -> None:
        """Execute a shell command."""
        print(f"\n\nExecuting command: {command} \n\n")
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("Error executing command:")
            print(result.stderr)
        else:
            print("Command executed successfully:")
            print(result.stdout)

        return result

    def run_job(self, slurm_command: str, singularity_command: str, bigcode_command: str) -> None:
        if not slurm_command or not singularity_command:
            return self.run_command(bigcode_command)
        
        """Run the job using SLURM and Singularity."""
        print(f"START TIME: {os.popen('date').read().strip()}\n")
        full_command = slurm_command + "--wrap='"
        load_command = "module purge && module load singularity && "

        # Note if we include the evaluation flag we must:
        # - Remove generation_only flag
        # - Substitute the load-generations-path by the save-generations-path
        # In normal scenarios this should not be needed, but we are using two images, one for inference, another to eval
        if self.args and self.args.generation_only is True:
            bigcode_command = bigcode_command[:-1] + " " + "--generation_only" + '"'
        elif self.args and self.args.evaluate_only is True:
            bigcode_command = bigcode_command.replace("--save_generations True", "--save_generations False")
            bigcode_command = bigcode_command.replace("--save_generations_path", "--load_generations_path")

        full_command += load_command + singularity_command + bigcode_command
        full_command += "'"

        self.logger.info(f"Running command: {full_command}")

        return self.run_command(full_command)


    def filter_model_config(self, config_dict: Dict, model_name: str) -> Dict:
        """
        Filters a configuration dictionary to include only the specified model's configuration.

        Parameters:
            - config_dict (dict): The original configuration dictionary containing multiple models.
            - model_name (str): The name of the model to filter for (e.g., "CodeLlama-70b-hf").

        Returns:
            - dict: A new dictionary with the same structure as the input, but containing only
                    the configuration for the specified model.
        """
        # Create a deep copy of the original dictionary to avoid modifying it
        filtered_config = config_dict.copy()

        # Find the model with the matching name
        matching_models = [
            model for model in config_dict["models"] if model["name"] == model_name
        ]

        if not matching_models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        # Replace the models list with only the matching model
        filtered_config["models"] = [matching_models[0]]

        return filtered_config

    def load_benchmark_config(self) -> List[Dict]:
        """
        Load the appropriate benchmark configuration based on args.
        Args:
            - None
        Returns:
            - benchmark_config: is a dictionary or a list of dictionaries
              containing the benchmark configuration.
        """
        if not self.args.run_all:
            config_path = (self.CONFIGS_DIR / self.args.benchmark).with_suffix(".yml")
            if not config_path.exists():
                self.logger.error(
                    f"Error: The benchmark config file '{config_path}' does not exist."
                )
                return []
            else:
                if self.args.model:
                    # finde into the benchmark config dictionary the name of model
                    flag_model = 0
                    benchmark_config = self.load_yaml_config(config_path)["benchmark"][
                        0
                    ]
                    for model in benchmark_config.get("models", []):
                        if model.get("name") == self.args.model:
                            flag_model += 1

                    if flag_model == 0:
                        self.logger.error(
                            f"Model '{self.args.model}' not found in the benchmark config file."
                        )
                        return []

                    return [self.filter_model_config(benchmark_config, self.args.model)]
                else:
                    return [self.load_yaml_config(config_path)["benchmark"][0]]
        else:
            configs = []
            for yaml_file in self.CONFIGS_DIR.glob("*.yml"):
                if yaml_file.name != "slurm.yml":
                    configs.extend(self.load_yaml_config(yaml_file)["benchmark"])
            return configs

    def load_jobs(self, benchmark_config: List[Dict]) -> None:
        """Execute the job with the given configuration.
        Parameters:
            - benchmark_config: A dictionary containing the benchmark configuration.
        Returns:

        """
        # check slurm configs
        filter_slurm = [
            model["slurm_config"]
            for item in benchmark_config
            for model in item.get("models", [])
            if "slurm_config" in model
        ]

        if filter_slurm and os.path.isfile(self.CONFIGS_DIR / "slurm.yml"):
            slurm_config_raw = self.load_yaml_config(self.CONFIGS_DIR / "slurm.yml")
            loader = SlurmConfigLoader(slurm_config_raw)

            # get all slurm configs
            slurm_configs_commands = loader.get_all_configs()
        else:
            slurm_configs_commands = {}

        # check singularity configs
        filter_singularity = [
            (item["task"], item["singularity_image"])
            for item in benchmark_config
            if "singularity_image" in item
        ]
        if filter_singularity:
            singularity_commands = self.build_singularity_commands(filter_singularity).get(_task)
        else:
            singularity_commands = None

        '''
        # check docker configs
        filter_docker = [
            (item["task"], item["docker_image"])
            for item in benchmark_config
            if "docker_image" in item
        ]
        if filter_docker:
            docker_commands = self.build_docker_commands(filter_docker)
        '''

        total_jobs = 0
        all_job_ids = []

        for benchmark in benchmark_config:
            _task = benchmark.get("task")
            # Build BigCode command
            builder = TurtleCommandBuilder(benchmark)

            for model in benchmark.get("models", []):
                # Process each temperature configuration
                for temp in model.get("temperature", []):
                    print("=" * 80)
                    print(
                        f" Process task: {_task} - Model: {model.get('name')} - Temp: {temp}"
                    )
                    print("=" * 80)

                    slurm_commands = slurm_configs_commands.get(
                        model.get("slurm_config")
                    )

                    # Build the command for task1 using accelerate
                    turtle_command = builder.build_command(model=model.get("name"))

                    # send job
                    result = self.run_job(
                        slurm_command=slurm_commands,
                        singularity_command=singularity_commands,
                        bigcode_command=turtle_command,
                    )
                    job_id = result.stdout.strip()

                    if job_id:
                        all_job_ids.append(job_id)
                        total_jobs += 1
                        print(f"    Job sending (T={temp}): ID {job_id}")
                    else:
                        print(f"    Error to send job: {result.stderr}")

        print("=" * 80)
        print("\n final report")
        print(f"Total jobs sending: {total_jobs}")
        print(f"IDs jobs: {', '.join(all_job_ids)}")

    def execute(self) -> None:
        """Execute the complete workflow."""
        # Load benchmark configuration
        benchmark_config = self.load_benchmark_config()

        if not benchmark_config:
            self.logger.error("Error: No benchmark configuration found.")
            return
        else:
            # Execute the job
            self.load_jobs(benchmark_config)
