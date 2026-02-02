# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

"""
NotSoTiny: Verilog Module Completion Benchmark using TinyTapeout Projects
NotSoTiny benchmark implemented by HPAI team at Barcelona Supercomputing Center (BSC).
"""

from datasets.load import load_dataset

import asyncio
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from turtle.metrics.code_eval import estimate_pass_at_k
from turtle.metrics.eval_notsotiny import eval_notsotiny_generation
from turtle.src.turtle_eval.base import TaskExtension


_CITATION = """
@misc{ghorab2025notsotinylargelivingbenchmark,
      title={NotSoTiny: A Large, Living Benchmark for RTL Code Generation}, 
      author={Razine Moundir Ghorab and Emanuele Parisi and Cristian Gutierrez-Gomez and Miquel Albert\'i-Binimelis and Miquel Moreto and Dario Garcia-Gasulla and Gokcen Kestor},
      year={2025},
      eprint={2512.20823},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={[https://arxiv.org/abs/2512.20823](https://arxiv.org/abs/2512.20823)}, 
}
"""


class NotSoTiny(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        super().__init__(stop_words=[], requires_execution=True)
        kwargs = kwargs.get("kwargs", {})
        self.model = kwargs.get("model")

        # Set-up basic params
        self.debug = False
        self.path_temporary_files = kwargs.get("path_temporary_files")
        self.load_generations_path = kwargs.get("load_generations_path", None)

        # Dataset
        self.dataset = load_dataset("HPAI-BSC/NotSoTiny-25-12", split="train")

        # Filter dataset by shuttle if --shuttle flag is provided
        self.shuttle = kwargs.get("shuttle", None)
        if self.shuttle:
            available_shuttles = set(self.dataset["shuttle_name"])
            if self.shuttle not in available_shuttles:
                raise ValueError(
                    f"Shuttle {self.shuttle} not available. Possible shuttles to choose from: {available_shuttles}"
                )
            self.dataset = self.dataset.filter(
                lambda x: x["shuttle_name"] == self.shuttle
            )

    def get_dataset(self):
        """
        Returns dataset for the task.
        """
        print(f"Loaded {len(self.dataset)} tasks from NotSoTiny dataset")
        return self.dataset

    def get_file(self, path: str):
        """Read file content."""
        with open(path, "r", encoding="utf-8", errors="ignore") as fd:
            contents = fd.read()
        return contents

    def get_prompt(self, doc):
        """
        Create prompt for LLM to complete the missing Verilog module.
        """
        conv = [
            {"role": "system", "content": doc["system_message"]},
            {"role": "user", "content": doc["prompt"]},
        ]
        return conv

    def get_reference(self, doc):
        """Get the golden solution content."""
        # Replaces golden module with prompted module from "prompt"
        # to get the full modules.v file with the golden module inplace
        prompt = doc["prompt"]
        golden_module = doc["golden_module"]

        # Extract the module name from the golden module (in case it's generic)
        module_match = re.search(r'module\s+(\w+)\s*\(', golden_module)
        if not module_match:
            raise ValueError("Could not find module name in golden module")

        module_name = module_match.group(1)
        pattern = rf'module\s+{module_name}\s*\([^)]*\);\s*//\s*>>>\s*Module Implementation Begin.*?//\s*<<<\s*Module Implementation End\s*endmodule'

        # Replace with the golden module
        golden_solution = re.sub(pattern, golden_module.strip(), prompt, flags=re.DOTALL)

        return golden_solution

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.

        Since our prompt asks for ONLY the code between the markers,
        we expect cleaner output, but still need to handle edge cases.
        """
        # For reasoning models, we keep only the final answer
        if "assistantfinal" in generation:  # gpt-oss
            delimiter = "assistantfinal"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
        elif "</think>" in generation:
            delimiter = "</think>"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
        else:
            reasoning = None

        task_id = self.dataset[idx]["task_id"]
        resp_content = generation.strip()

        # Remove markdown code blocks if present
        if "```verilog" in resp_content:
            resp_content = resp_content.replace("```verilog", "```")
        if "```" in resp_content:
            pattern = re.compile(r"```(?:\w+)?\s*([\s\S]+?)\s*```")
            match = pattern.search(resp_content)
            if match:
                resp_content = match.group(1).strip()

        # Since we asked for ONLY the implementation code, we should get clean output
        # But still need to handle some edge cases

        # Remove any accidental inclusion of the markers themselves
        for pattern in [
            r"//\s*>>>\s*Module Implementation Begin.*?\n?",
            r"//\s*<<<\s*Module Implementation End.*?\n?",
        ]:
            resp_content = re.sub(
                pattern,
                "",
                resp_content,
                flags=re.IGNORECASE,
            )

        # Remove any module declarations or endmodule statements (shouldn't be there)
        lines = resp_content.split("\n")
        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()
            # Skip module declarations and endmodule statements
            if (
                line_stripped.startswith("module ")
                or line_stripped == "endmodule"
                or line_stripped.startswith("endmodule")
            ):
                continue
            filtered_lines.append(line)

        resp_content = "\n".join(filtered_lines).strip()

        # Now we need to create the complete module by inserting this into the prompt
        task_content = self.dataset[idx]["prompt"]

        # Replace the placeholder region with the generated implementation
        placeholder_pattern = r"//\s*>>>\s*Module Implementation Begin.*?//\s*<<<\s*Module Implementation End"

        complete_module = re.sub(
            placeholder_pattern,
            lambda m: resp_content,
            task_content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return {
            "prompt": self.datadet[idx]["prompt"],
            "golden_module": self.datadet[idx]["golden_module"],
            "reasoning": reasoning,
            "generation": complete_module,  # Complete module with implementation filled in
        }

    def _evaluate_stx_fnc(
        self, generation: str, golden_solution: str, task_id: str, id: str, top_module_name: str
    ) -> dict:
        with (
            tempfile.NamedTemporaryFile(
                suffix=".v", delete=True, dir=self.path_temporary_files
            ) as f_gen,
            tempfile.NamedTemporaryFile(
                suffix=".v", delete=True, dir=self.path_temporary_files
            ) as f_sol,
        ):
            f_gen.write(generation.encode("utf-8"))
            f_gen.flush()

            f_sol.write(golden_solution.encode("utf-8"))
            f_sol.flush()

            result = eval_notsotiny_generation(
                Path(f_gen.name), Path(f_sol.name), task_id, id, top_module_name, self.debug
            )
        return result

    def _compute_pass_k(
        self, correct: np.ndarray, total: np.ndarray, ks: list[int]
    ) -> dict:
        results = {}
        for k in ks:
            if (total >= k).all():
                results[f"pass@{k}"] = round(
                    estimate_pass_at_k(total, correct, k).mean() * 100, 2
                )
        return results

    async def _evaluate_stx_async(
        self,
        generation: str,
        golden_solution: str,
        task_id: str,
        id: Optional[str],
        top_module_name: str,
    ) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._evaluate_stx_fnc, generation, golden_solution, task_id, id, top_module_name
        )

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        """
        reports = defaultdict(dict)  # This will store detailed per-task results

        async def _run_all_flat():
            # Create tasks with task_id tracking
            flat_tasks = []
            task_metadata = []  # Store task_id for each evaluation

            for task_idx, (gens, refs) in enumerate(zip(generations, references)):
                if task_idx >= len(self.dataset):
                    print(
                        f"Warning: task_idx {task_idx} exceeds dataset size {len(self.dataset)}"
                    )
                    continue

                task_id = self.dataset[task_idx]["task_id"]
                top_module_name = self.dataset[task_idx]["top_module_name"]
                for gen_idx, (g, golden_solution) in enumerate(zip(gens, refs)):
                    flat_tasks.append(
                        self._evaluate_stx_async(
                            g["generation"],
                            golden_solution,
                            task_id=task_id,
                            id=f"{task_id}_gen{gen_idx}",
                            top_module_name=top_module_name,
                        )
                    )
                    task_metadata.append(
                        {
                            "task_idx": task_idx,
                            "task_id": task_id,
                            "gen_idx": gen_idx,
                        }
                    )

            # Run all evaluations
            print(
                f"Running {len(flat_tasks)} evaluations across {len(generations)} tasks..."
            )
            results = await asyncio.gather(*flat_tasks)

            # Store detailed results and compute aggregates
            correct_syntax, correct_equiv, correct_func, total = [], [], [], []
            idx = 0

            for task_idx, gens in enumerate(generations):
                # Safely get task info with bounds checking
                if task_idx >= len(self.dataset):
                    print(f"Warning: Skipping task_idx {task_idx} - out of bounds")
                    continue

                task_id = self.dataset[task_idx]["task_id"]
                chunk = results[idx : idx + len(gens)]
                idx += len(gens)

                # Store detailed results for each generation of this task
                reports[task_id] = {
                    "project_name": self.dataset[task_idx]["project_name"],
                    "module_name": self.dataset[task_idx]["module_name"],
                    "num_generations": len(gens),
                    "generations": [],
                }

                for gen_idx, result in enumerate(chunk):
                    generation_result = {
                        "generation_idx": gen_idx,
                        "syntax_passed": result["syntax_passed"],
                        "equiv_passed": result["equiv_passed"],
                        "passfail": result["passfail"],
                        "syntax_error": result.get("syntax_error", ""),
                        "equiv_error": result.get("equiv_error", ""),
                        "top_module": result.get("top_module", ""),
                        "eqy_return_code": result.get("eqy_return_code", None),
                        "equiv_method": result.get("equiv_method", ""),
                        "warnings": result.get("warnings", []),
                    }
                    reports[task_id]["generations"].append(generation_result)

                # Compute aggregates for this task
                correct_syntax.append(sum(int(r["syntax_passed"]) for r in chunk))
                correct_equiv.append(sum(int(r["equiv_passed"]) for r in chunk))
                total.append(len(gens))

                # Add summary statistics to the task report
                reports[task_id]["summary"] = {
                    "syntax_pass_count": correct_syntax[-1],
                    "equiv_pass_count": correct_equiv[-1],
                    "total_generations": total[-1],
                    "syntax_pass_rate": f"{correct_syntax[-1] / total[-1] * 100:.2f}%"
                    if total[-1] > 0
                    else "0.00%",
                    "equiv_pass_rate": f"{correct_equiv[-1] / total[-1] * 100:.2f}%"
                    if total[-1] > 0
                    else "0.00%",
                }

            return correct_syntax, correct_equiv, correct_func, total

        correct_syntax, correct_equiv, correct_func, total = asyncio.run(
            _run_all_flat()
        )

        # Save detailed report
        if self.load_generations_path:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path)) / "detailed_report.json"

            # Create a more structured report
            final_report = {
                "benchmark": "NotSoTiny",
                "model": self.model,
                "total_tasks": len(self.dataset),
                "total_generations": sum(total),
                "overall_summary": {
                    "syntax_pass_rate": f"{sum(correct_syntax) / sum(total) * 100:.2f}%"
                    if sum(total) > 0
                    else "0.00%",
                    "equiv_pass_rate": f"{sum(correct_equiv) / sum(total) * 100:.2f}%"
                    if sum(total) > 0
                    else "0.00%",
                    "func_pass_rate": f"{sum(correct_func) / sum(total) * 100:.2f}%"
                    if sum(total) > 0
                    else "0.00%",
                },
                "per_task_results": reports,
            }

            with open(output_dir, "w") as f:
                json.dump(final_report, f, indent=2)

            print(f"\n{'=' * 80}")
            print(f"Detailed report saved to: {output_dir}")
            print(f"{'=' * 80}\n")

            # Also save a CSV summary for easy analysis
            csv_path = Path(os.path.dirname(json_path)) / "summary_report.csv"
            csv_data = []
            for task_id, task_report in reports.items():
                csv_data.append(
                    {
                        "task_id": task_id,
                        "project_name": task_report["project_name"],
                        "module_name": task_report["module_name"],
                        "num_generations": task_report["num_generations"],
                        "syntax_pass_count": task_report["summary"][
                            "syntax_pass_count"
                        ],
                        "equiv_pass_count": task_report["summary"]["equiv_pass_count"],
                        "func_pass_count": task_report["summary"]["func_pass_count"],
                        "syntax_pass_rate": task_report["summary"]["syntax_pass_rate"],
                        "equiv_pass_rate": task_report["summary"]["equiv_pass_rate"],
                        "func_pass_rate": task_report["summary"]["func_pass_rate"],
                    }
                )

            if csv_data:  # Only create CSV if we have data
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                print(f"Summary CSV saved to: {csv_path}\n")
        else:
            print(
                "Warning: load_generations_path not set, skipping detailed report generation"
            )

        ks = [1, 5, 20]
        return {
            "syntax": self._compute_pass_k(
                np.array(correct_syntax), np.array(total), ks
            ),
            "equiv": self._compute_pass_k(np.array(correct_equiv), np.array(total), ks),
            "func": self._compute_pass_k(np.array(correct_func), np.array(total), ks),
        }
