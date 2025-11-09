import logging
import os
import re
import subprocess
import tempfile
import time
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
        """Initialize the WorkflowExecutor.
        Parameters:
            - args: Dictionary of command line arguments
            - logger: Configured logger instance
        Returns:
            None
        """
        self.args = args
        self.file_handler = FileHandler()
        self.registry_updater = RegistryUpdater()
        self.CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
        self.logger = logger
        self.slurm_loader = None

        # Multi-node execution state
        self.head_node_ip = None
        self.ray_port = None
        self.vllm_port = None
        self.running_processes = []

    def load_yaml_config(self, file_path: Path) -> Dict:
        """Load a YAML configuration file.
        Parameters:
            - file_path: Path to the YAML file.
        Returns:
            - config: A dictionary containing the loaded configuration.
        """
        return self.file_handler.load_yaml(file_path)

    def build_singularity_commands(self, task_image_list: list[tuple[str, str]]) -> dict[str, str]:
        """Build a dictionary of Singularity commands for each task.

        Args:
            task_image_list: List of tuples where each tuple contains (task_name, image_path)

        Returns:
            Dictionary mapping task names to their corresponding Singularity commands
        """
        commands = {}
        for task_name, image_path in task_image_list:
            commands[task_name] = (
                f"VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 singularity exec --nv {image_path} bash -c "
            )
        return commands

    def build_docker_commands(self, task_image_list: list[tuple[str, str]]) -> dict[str, str]:
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

    # Methods for multi-node execution
    def _setup_environment(self, slurm_config: str = None) -> None:
        """Set up environment variables for distributed execution."""
        # Essential environment variables from bash script
        enviroment_vars = self.slurm_loader.get_env_vars(slurm_config)

        # Apply environment variables
        eviroment = {}
        for key, value in enviroment_vars.items():
            eviroment[key] = value

        return eviroment

    def parse_sbatch_string(self, slurm_command: str) -> Dict[str, str]:
        """Parse a string containing sbatch parameters into a dictionary.
        Parameters:
            - slurm_command: String containing sbatch parameters
        Returns:
            - Dictionary with parameter names as keys and their values
        """
        # Delete 'sbatch' from the beginning of the string if it exists
        if slurm_command.startswith("sbatch "):
            sbatch_str = slurm_command[7:]

        # Split the string into parts
        parts = slurm_command.split()

        result = {}

        for part in parts:
            # check if the part starts with '--' (indicating a parameter)
            if part.startswith("--"):
                # split the part into key and value
                key_value = part[2:].split("=", 1)
                key = key_value[0]
                # If there is no value (e.g., --exclusive), assign None
                value = key_value[1] if len(key_value) > 1 else None
                result[key] = value

        return result

    def build_multinode_commands(self, dictionary: Dict[str, str]) -> str:
        """Build commands for multi-node execution.
        Parameters:
            - dictionary: Dictionary containing sbatch parameters
        Returns:
            - rendered_script: Rendered script for multi-node execution
        """
        from jinja2 import Environment, FileSystemLoader

        base = Path(__file__).parent.parent
        path_template = base / "template"

        benchmark = dictionary["task"]
        model = dictionary["model_name"]

        # generate_only flag
        if self.args and self.args.generation_only is True:
            dictionary["turtle_commands"] = (
                dictionary["turtle_commands"][:-1] + " " + "--generation_only" + '"'
            )

        env = Environment(
            loader=FileSystemLoader(str(path_template)),
            keep_trailing_newline=True,
        )

        template_dir = path_template / benchmark
        template = env.get_template("multinode.sh.j2")
        rendered = template.render(dictionary)

        output_dir = template_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{model}.sh"
        output_path.write_text(rendered, encoding="utf-8")

        return rendered

    # End of multi-node execution

    def run_command(self, command: str) -> None:
        """Execute a shell command safely.
        Parameters:
            - command: Command string to execute
        Returns:
            - result: CompletedProcess object containing the result of the command
        """
        # print(command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(f"Command failed (exit code {result.returncode})")
            else:
                self.logger.info("Command executed successfully")

            return result
        except Exception as e:
            self.logger.exception(f"Exception while executing command: {str(e)}")
            # Create a dummy result to return in case of exception
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=str(e))

    def run_job(
        self,
        slurm_command: str,
        slurm_confgs: str,
        singularity_command: str,
        turtle_command: str,
        task: str = None,
        singularity_image: str = None,
        model_path: str = None,
        model_name: str = None,
        type_job: str = None,
        temperature: str = None,
        benchmark_config: List[Dict] = None,
        turtle_commands: str = None,
    ) -> subprocess.CompletedProcess:
        """Run job using either SLURM or direct multi-node execution.
        Parameters:
            - slurm_command: SLURM sbatch command
            - turtle_command: Task command to execute
            - task: Task name (multi-node only)
            - singularity_image: Singularity image path (multi-node only)
            - model_path: Model directory path (multi-node only)
        Returns:
            - CompletedProcess with execution results
        """
        self.logger.info(f"JOB START TIME: {time.ctime()}")

        if slurm_confgs == "api":
            self.logger.info("API execution mode detected - running locally without SLURM")
            clean_command = turtle_command.strip('"')
            return self.run_command(clean_command)

        if self.args and self.args.generation_only is True:
            turtle_command = turtle_command[:-1] + " " + "--generation_only" + '"'

        if slurm_command != "":
            modified_slurm_command = slurm_command.replace("--cpus-per-task", "--cpus_per_task")
            modified_slurm_command = modified_slurm_command.replace("--ntasks-per-node", "--ntasks_per_node")
            dictionary = self.parse_sbatch_string(modified_slurm_command)
            # Add task, singularity image, and model path to the dictionary
            dictionary["slurm_enabled"] = True
        else:
            dictionary = {}

        dictionary["task"] = task
        if singularity_image:
            dictionary["singularity_enabled"] = True
            dictionary["singularity_image"] = singularity_image
            if (self.args and self.args.evaluation_only is True) and benchmark_config[0].get(
                "evaluation_image", False
            ):
                singularity_command = singularity_command.replace(
                    singularity_image, benchmark_config[0].get("evaluation_image")
                )
                dictionary["singularity_image"] = benchmark_config[0].get("evaluation_image")
                if dictionary.get("slurm_enabled", False):
                    slurm_command = re.sub(r"--nodes=\d+", "--nodes=1", slurm_command)

        dictionary["model_name"] = model_name
        dictionary["model_path"] = model_path
        dictionary["temperature"] = temperature
        dictionary["metric_output_path"] = benchmark_config[0].get("metric_output_path")

        string1, string2 = turtle_commands.split("--model", 1)
        new_turtle_commands = (
            f"{string1.strip()} --ip ${{head_node_ip}} --port ${{vllm_port}} --model{string2}"
        )

        dictionary["turtle_commands"] = new_turtle_commands

        # Add environment variables to the dictionary
        dictionary.update(self._setup_environment(slurm_confgs))

        # The idea is that when we run an evaluation only command, there is no need to go for a multi-node setup
        if "multi-node" in type_job and (self.args and self.args.evaluation_only is not True):
            # 1. Fill jinja2 template with
            multinode_command = self.build_multinode_commands(dictionary)

            # 2. Create a temporary file to store the script
            path_template = "template"
            temp_folder = Path(__file__).parent.parent / path_template

            # 2. Create a temporary file to store the script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", dir=temp_folder, delete=False) as f:
                f.write(multinode_command)
                script_path = f.name

            # 3. Make the script executable
            os.chmod(script_path, 0o755)

            full_cmd = (
                # f"{slurm_command} --wrap='"
                f"sbatch {script_path}'"
            )
        else:
            # Note if we include the evaluation flag we must:
            # - Remove generation_only flag
            # - Substitute the load-generations-path by the save-generations-path
            # - Perform evaluation on GPP
            # In normal scenarios this should not be needed, but we are using two images, one for inference, another to eval
            if self.args and self.args.evaluation_only is True:
                slurm_command = re.sub(r"--qos=\S+", "--qos=gp_bsccs", slurm_command)
                slurm_command = re.sub(r"--gres=gpu:\d+", "", slurm_command)
                slurm_command = re.sub(r"#SBATCH\s+--exclusive", "", slurm_command)
                slurm_command = re.sub(
                    r"--cpus-per-task=\d+",
                    "--cpus-per-task=80",
                    slurm_command,
                )
                turtle_command = turtle_command.replace("--save_generations True", "--save_generations False")
                turtle_command = turtle_command.replace("--save_generations_path", "--load_generations_path")

            # Single-node SLURM execution
            full_cmd = (
                f"{slurm_command} --wrap="
                f"'module purge && module load singularity && "
                f"{singularity_command} {turtle_command}'"
            )

        return self.run_command(full_cmd)

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
        matching_models = [model for model in config_dict["models"] if model["name"] == model_name]

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
                self.logger.error(f"Error: The benchmark config file '{config_path}' does not exist.")
                return []
            else:
                if self.args.model:
                    # finde into the benchmark config dictionary the name of model
                    flag_model = 0
                    benchmark_config = self.load_yaml_config(config_path)["benchmark"][0]
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
            self.slurm_loader = SlurmConfigLoader(slurm_config_raw)

            slurm_configs_commands = self.slurm_loader.get_all_configs()
        else:
            slurm_configs_commands = {}

        # check singularity configs
        filter_singularity = [
            (item["task"], item["singularity_image"])
            for item in benchmark_config
            if "singularity_image" in item
        ]
        if filter_singularity:
            singularity_commands = self.build_singularity_commands(filter_singularity)

        # check docker configs
        filter_docker = [
            (item["task"], item["docker_image"]) for item in benchmark_config if "docker_image" in item
        ]
        if filter_docker:
            docker_commands = self.build_docker_commands(filter_docker)

        total_jobs = 0
        all_job_ids = []

        for benchmark in benchmark_config:
            _task = benchmark.get("task")
            # Build BigCode command
            builder = TurtleCommandBuilder(benchmark)

            for model in benchmark.get("models", []):
                # Process each temperature configuration
                for temp in model.get("temperature", []):
                    self.logger.info("=" * 80)
                    ttitle = self.aligns_text(
                        f"Process task: {_task} - Model: {model.get('name')} - Temp: {temp}"
                    )
                    self.logger.info(ttitle)
                    self.logger.info("=" * 80)

                    # If slurm config command dictionary is not empty
                    if slurm_configs_commands:
                        slurm_commands = slurm_configs_commands.get(model.get("slurm_config"))
                    else:
                        slurm_commands = ""

                    # Build the command required for turtle
                    turtle_command = builder.build_command(model=model.get("name"))

                    # Add generation_only flag if present in args
                    if self.args and self.args.generation_only:
                        turtle_command = turtle_command.rstrip('"') + ' --generation_only"'

                    # send job
                    result = self.run_job(
                        slurm_command=slurm_commands,
                        slurm_confgs=model.get("slurm_config"),
                        singularity_command=singularity_commands.get(_task),
                        turtle_command=turtle_command,
                        task=benchmark.get("task"),
                        singularity_image=benchmark.get("singularity_image"),
                        model_path=benchmark.get("path_model"),
                        model_name=model.get("name"),
                        type_job=model.get("slurm_config"),
                        temperature=temp,
                        benchmark_config=benchmark_config,
                        turtle_commands=turtle_command,
                    )
                    job_id = result.stdout.strip()

                    if job_id:
                        all_job_ids.append(job_id)
                        total_jobs += 1
                        self.logger.info(f"Job sending (T={temp}): ID {job_id}")
                    else:
                        self.logger.error(f"Error to send job: {result.stderr}")

        self.logger.info("=" * 80)
        tittle = self.aligns_text("FINAL REPORT")
        self.logger.info(tittle)
        self.logger.info(f"Total jobs submitted: {total_jobs}")
        self.logger.info("-" * 80)

        if all_job_ids:
            # Create a table-like presentation for job IDs
            self.logger.info("+" + "-" * 30 + "+")
            subttitle = self.aligns_text("Job ID", 28)
            self.logger.info("| " + subttitle.ljust(28) + " |")
            self.logger.info("+" + "-" * 30 + "+")

            for job_id in all_job_ids:
                self.logger.info("| " + job_id.ljust(28) + " |")

            self.logger.info("+" + "-" * 30 + "+")
            self.logger.info("=" * 80)
        else:
            self.logger.info("No jobs were successfully submitted.")

    def aligns_text(self, tittle: str = "", weight: int = 80) -> str:
        """Center the text in a string of a given width.
        Parameters:
            - tittle: The text to center
            - weight: The width of the string
        Returns:
            - centered_text: The centered text
        """
        # If the title is longer than the weight, return it as is
        if len(tittle) >= weight:
            return tittle

        # find the number of white spaces to add
        white_spaces = (weight - len(tittle)) // 2
        return " " * white_spaces + tittle

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
