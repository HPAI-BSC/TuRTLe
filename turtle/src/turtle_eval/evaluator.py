import json
import os
import warnings

from src.utils.task_updater import TaskUpdater

# from importlib import import_module
from .generation import get_generations

_WARNING = """
This is a warning message indicating that code execution is not allowed.

To enable code execution, set the `allow_code_execution` argument to True.
If you are using a pre-trained model, it is recommended to use the `--allow_code_execution` flag.
"""


class EvaluatorAdapter:
    """Adaptador class to modified evaluator for LCase."""

    def __init__(self, model, client, tokenizer, args):
        """
        Modified constructor to use the new evaluator,
        without use accelerator
        """

        self.model = model
        self.client = client
        self.tokenizer = tokenizer
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        """
        Simplified version without `intermediate_generations`.
        """
        # task = import_module('eval_harness.tasks').get_task(task_name, self.args)
        task = TaskUpdater().get_task(task_name, self.args)
        dataset = task.get_dataset()
        # n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start if self.args.limit else len(dataset))
        n_tasks = (
            min(self.args.limit, len(dataset) - self.args.limit_start)
            if self.args.limit
            else len(dataset) - self.args.limit_start
        )

        if not self.args.limit:
            n_tasks -= self.args.limit_start

        references = [
            task.get_reference(dataset[i])
            for i in range(self.args.limit_start, self.args.limit_start + n_tasks)
        ]

        generations = get_generations(
            task, dataset, self.model, self.client, self.tokenizer, n_tasks, self.args
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name):
        """Simplified version of evaluate method."""
        # task = import_module('eval_harness.tasks').get_task(task_name, self.args)
        # if task.requires_execution and not self.allow_code_execution:
        #    raise ValueError(_WARNING)

        task = TaskUpdater().get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name)

        if not self.args.load_generations_path:
            self._save_json_files(
                generations,
                references,
                self.args.save_generations_path,
                self.args.save_references_path,
            )

        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        return task.process_results(generations, references)

    def _save_json_files(self, generations, references, gen_path, ref_path):
        """
        Aditional json file saving method.
        This method is used to save the generations and references
        to the specified paths.
        """
        if self.args.save_generations:
            os.makedirs(os.path.dirname(gen_path), mode=0o755, exist_ok=True)
            with open(gen_path, "w+") as f:
                json.dump(generations, f)
                print(f"Generations saved at {gen_path}")
        if self.args.save_references:
            os.makedirs(os.path.dirname(ref_path), mode=0o755, exist_ok=True)
            with open(ref_path, "w+") as f:
                json.dump(references, f)
                print(f"References saved at {ref_path}")


# class LCaseEvaluatorProxy:
#     """Proxy that decides which evaluator to use."""

#     def __new__(cls, use_modified=True, *args, **kwargs):
#         if use_modified:
#             return EvaluatorAdapter(*args, **kwargs)

# how to use the factory
# def get_evaluator(model, tokenizer, args):
#     """Factoy that return the evaluator."""
#     return LCaseEvaluatorProxy(model, tokenizer, args)
