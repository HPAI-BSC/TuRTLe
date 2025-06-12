import inspect
import sys
from importlib import import_module
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional


class TaskUpdater:
    """
    A class to extend and update the task registry from an external library without modifying it directly.

    Attributes:
        original_module: Reference to the original tasks module
        extended_registry: Combined task registry with original and new tasks
        all_tasks: Sorted list of all task names
    """

    def __init__(self) -> None:
        """Initialize the TaskUpdater and load the original module.
        Parameters:
            - FileHandler: Optional file handler for loading YAML files
        Args:
            - None
        """
        try:
            self.original_module = import_module("bigcode_eval.tasks")
            self.original_registry = getattr(self.original_module, "TASK_REGISTRY", {})
            self.original_all_tasks = getattr(self.original_module, "ALL_TASKS", [])
            self.original_get_task = getattr(self.original_module, "get_task", None)
        except ModuleNotFoundError:
            print("Warning: bigcode_eval.tasks not found. Skipping original registry.")
            self.original_module = None
            self.original_registry = {}
            self.original_all_tasks = []
            self.original_get_task = None

        self.extended_registry = {}
        self.all_tasks = []

        self._load_new_modules()
        self._create_extended_registry()
        self._update_all_tasks()

    def _load_new_modules(self):
        """Dynamically import the new task modules."""
        # Add the src directory to sys.path
        src_path = str(Path(__file__).parent.parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Importar usando la ruta completa desde el directorio src
            self.verilog_eval = import_module("tasks.verilog_eval")
            self.rtllm = import_module("tasks.rtllm")
            self.rtlrepo = import_module("tasks.rtlrepo")
            self.verigen = import_module("tasks.vgen")
        except ImportError as e:
            print(f"Error importing modules: {e}")
            print(f"Current sys.path: {sys.path}")
            raise

    def _create_extended_registry(self):
        """Combine original and new tasks into extended registry."""
        self.extended_registry = {
            **self.original_registry,
            "verilog_eval_cc": self.verilog_eval.VerilogEvalCodeComplete,
            "verilog_eval_rtl": self.verilog_eval.VerilogEvalRTLToSpecification,
            "rtllm": self.rtllm.RTLLM,
            "RTLRepo": self.rtlrepo.RTLRepo,
            "VeriGen": self.verigen.VeriGen,
        }

    def _update_all_tasks(self):
        """Update the list of all task names."""
        self.all_tasks = sorted(list(self.extended_registry.keys()))

    def get_task(self, task_name: str, args: Optional[Any] = None) -> Any:
        """
        Extended version of get_task that handles additional parameters.

        Args:
            task_name: Name of the task to instantiate
            args: Optional object containing task parameters

        Returns:
            An instance of the requested task

        Raises:
            KeyError: If the task name is not found
        """
        try:
            task_class = self.extended_registry[task_name]
            args_dict = vars(args) if args is not None else {}

            # Si la clase espera {'kwargs': {todos_los_parametros}}
            return task_class(kwargs=args_dict)

        except Exception as e:
            print(f"Error creating {task_name}: {str(e)}")
            raise

    def apply_updates(self):
        """Apply the updates to the original module."""

        if self.original_module:
            setattr(self.original_module, "ALL_TASKS", self.all_tasks)
            setattr(self.original_module, "TASK_REGISTRY", self.extended_registry)
            setattr(self.original_module, "get_task", self.get_task)

    @property
    def ALL_TASKS(self) -> List[str]:
        """Get the updated list of all tasks."""
        return self.all_tasks

    @property
    def TASK_REGISTRY(self) -> Dict[str, Any]:
        """Get the extended task registry."""
        return self.extended_registry
