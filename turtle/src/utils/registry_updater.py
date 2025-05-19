import re
from pathlib import Path
from typing import Dict

class RegistryUpdater:
    def update(self, project: str, benchmark_config: Dict) -> None:
        """Update the task registry and imports."""
        init_file_path = Path("bigcode-evaluation-harness") / "bigcode_eval" / "tasks" / "__init__.py"
        
        if not init_file_path.exists():
            raise FileNotFoundError(f"File '{init_file_path}' does not exist.")

        content = self._read_file(init_file_path)
        updated_content = self._update_registry(content, project, benchmark_config)
        updated_content = self._update_imports(updated_content, project)
        
        self._write_file(init_file_path, updated_content)
        print(f"Updated TASK_REGISTRY and import statements in {init_file_path}.")

    def _read_file(self, path: Path) -> str:
        """Read file content."""
        with open(path, "r") as file:
            return file.read()

    def _write_file(self, path: Path, content: str) -> None:
        """Write content to file."""
        with open(path, "w") as file:
            file.write(content)

    def _update_registry(self, content: str, project: str, benchmark_config: Dict) -> str:
        """Update the TASK_REGISTRY content."""
        pattern = re.compile(r"TASK_REGISTRY\s*=\s*\{([\s\S]*?)\}", re.MULTILINE)
        match = pattern.search(content)
        if not match:
            raise ValueError("TASK_REGISTRY not found in __init__.py.")

        registry_content = match.group(1).strip()
        registry_content = self._add_new_tasks(registry_content, project, benchmark_config)
        
        return content[:match.start(1)] + registry_content + content[match.end(1):]

    def _update_imports(self, content: str, project: str) -> str:
        """Update the import statements."""
        pattern = re.compile(r"from\s+\.\s+import\s+\(([\s\S]*?)\)", re.MULTILINE)
        match = pattern.search(content)
        if not match:
            raise ValueError("Import statements not found in __init__.py.")

        import_content = match.group(1).strip()
        import_content = self._add_new_imports(import_content, project)
        
        return content[:match.start(1)] + import_content + content[match.end(1):]

    def _add_new_tasks(self, registry_content: str, project: str, benchmark_config: Dict) -> str:
        """Add new tasks to the registry content."""
        task_dir = Path("task") / project
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory '{task_dir}' does not exist.")

        for task_file in task_dir.glob("*.py"):
            module_name = task_file.stem
            with open(task_file, "r") as file:
                file_content = file.read()
            class_names = re.findall(r"class\s+(\w+)\s*\(", file_content)

            for task in benchmark_config["tasks"]:
                for class_name in class_names:
                    task_class = f"{module_name}.{class_name}"
                    new_entry = f'"{task["name"]}": {task_class},'
                    if new_entry not in registry_content:
                        registry_content += f"\n    {new_entry}"

        return registry_content

    def _add_new_imports(self, import_content: str, project: str) -> str:
        """Add new imports to the import content."""
        task_dir = Path("task") / project
        for task_file in task_dir.glob("*.py"):
            module_name = task_file.stem
            if module_name not in import_content:
                import_content += f",\n    {module_name}"
        return import_content