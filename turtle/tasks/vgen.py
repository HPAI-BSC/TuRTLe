# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

"""
Benchmarking Large Language Models for Automated Verilog RTL Code Generation
https://arxiv.org/pdf/2212.11140
VeriGen benchmark implemented by HPAI team at Barcelona Supercomputing Center (BSC).
Homepage: https://github.com/shailja-thakur/VGen
"""
import json
import os
import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from metrics.code_eval import estimate_pass_at_k
from metrics.eval_verilog import eval_script_verigen
from metrics.openlane_unified import (
    create_problem_structure,
    run_openlane_for_generation,
)
from metrics.ppa_score import compute_ppa_score
from src.turtle_eval.base import TaskExtension
from transformers import AutoTokenizer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

_CITATION = """
@misc{https://doi.org/10.48550/arxiv.2212.11140,
  doi = {10.48550/ARXIV.2212.11140},
  url = {https://arxiv.org/abs/2212.11140},
  author = {Thakur, Shailja and Ahmad, Baleegh and Fan, Zhenxing and Pearce, Hammond and Tan, Benjamin and Karri, Ramesh and Dolan-Gavitt, Brendan and Garg, Siddharth},
  title = {Benchmarking Large Language Models for Automated Verilog RTL Code Generation},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""


class VeriGen(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        super().__init__(
            stop_words=["endmodule", "endmodulemodule"], requires_execution=False
        )

        # get the arguments
        kwargs = kwargs.get("kwargs", {})
        self.model = kwargs.get("model")
        self.simulator = kwargs.get("simulator", "icarus")
        self.path_model = kwargs.get("path_model")
        self.prompt = kwargs.get("prompt")
        few_shot = kwargs.get("few_shot")
        path_data_benchmark = kwargs.get("path_data_benchmark")
        path_dataset_test = kwargs.get("path_dataset_test")
        self.path_temporary_files = kwargs.get("path_temporary_files")
        self.metric_output_path = kwargs.get("metric_output_path")

        # Set-up basic params
        if not self.prompt:
            self.prompt = "default"
        self.debug = True
        self.task_name = "verigen"
        assert few_shot >= 0 and few_shot <= 4, "Few shot supported range is 0-4."
        self.examples = few_shot

        # Make sure we have access to the dataset and the repo
        path_verigen_repo = path_data_benchmark
        if not os.path.exists(path_verigen_repo):
            raise ValueError(
                "Path to `verigen` repo not found.\n`{path_verigen_repo}` does not exists."
            )
        self.path_dataset = os.path.join(path_verigen_repo, "Benchmark")
        # self.path_examples = os.path.join(path_verigen_repo, "scripts")
        self.dataset = load_from_disk(path_dataset_test)
        # Only process prompts with intermediate level of detail
        self.dataset = [
            entry for entry in self.dataset if "prompt2" in entry["task_id"]
        ]

        # setup directories for (later-on) computing the PPA
        self.load_generations_path = kwargs.get("load_generations_path", None)
        print(f"self.load_generations_path: {self.load_generations_path}")
        if self.load_generations_path is not None:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path) + "/")
            _ = create_problem_structure(output_dir, json_path)
            self.base_gen_dir = output_dir
        else:
            self.base_gen_dir = None

    # TODO: Delete this, just a helper function that will print contents generated for debug purposes
    def _printHelper(self, title: str, content: str):
        if self.debug:
            print(
                "\n"
                + "-" * 30
                + title
                + "-" * 30
                + "\n"
                + content
                + "\n"
                + "-" * 70
                + "\n"
            )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        if self.examples == 0:
            return ""
        filename = f"verigen-example-prefix_{self.task_name}_{self.examples}-shot.txt"
        file = os.path.join(self.path_examples, filename)
        assert (
            os.path.exists(file) == True
        ), f"Fewshot example for n_shot = {self.examples} not found or possible duplicate."
        with open(file) as fd:
            contents = fd.read()
        return "\n" + contents

    def get_path(self, task_id: str, suffix: str = ""):
        # Split task_id in folder_name and file_base
        try:
            folder_name, file_base = task_id.split("-", 1)
        except ValueError:
            raise ValueError(
                f"Invalid task_id format: {task_id}. Expected format is 'folder-file'."
            )

        # Build the full path to the folder
        folder_path = os.path.join(self.path_dataset, folder_name)

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"The folder '{folder_name}' does not exist in the dataset path."
            )

        # Buiild the patern of search according to whether there is a suffix or not
        if suffix:
            # If there is a suffix, we search for files that start with that suffix and have the .v extension
            regex = re.compile(rf"^{re.escape(suffix)}_.+\.v$")
        else:
            # If there is no suffix, we search for files that have the .v extension
            regex = re.compile(rf"^{re.escape(file_base)}\.v$")

        # find the file that matches the pattern
        matching_files = [f for f in os.listdir(folder_path) if regex.match(f)]

        # Check that exists only one file that matches the pattern
        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No file found matching the pattern in folder '{folder_name}'."
            )
        elif len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found matching the pattern in folder '{folder_name}': {matching_files}"
            )

        # build and return the full path to the file
        file_path = os.path.join(folder_path, matching_files[0])
        return file_path

    def get_file(self, task_id: str, suffix: str):
        file = self.get_path(task_id, suffix)
        with open(file) as fd:
            contents = fd.read()
        return contents

    def get_prompt(self, doc):
        ret = self.get_file(doc["task_id"], "")

        # Only apply chat template for reasoning models
        if self.prompt == "reasoning":
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model, trust_remote_code=True
                )
                conversation = [
                    {"role": "user", "content": ret},
                ]
                ret = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                ret += "<think>\n"
                # ret += "\n<think></think>\n"
            except ValueError:
                print(f"Warning: {self.model} does not has a tokenizer template.")

        self._printHelper("PROMPT", ret)
        return ret

    def get_reference(self, doc):
        return doc["canonical_solution"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        # For reasoning models, we keep only the final answer
        if "</think>" in generation:
            self._printHelper("STRIP GENERATION", generation)
            delimiter = "</think>"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
            self._printHelper("REASONING", reasoning)
        else:
            reasoning = "None"

        prompt = self.get_prompt(self.dataset[idx])
        task_id = self.dataset[idx]["task_id"]
        raw_prompt = self.get_file(task_id, "")
        self._printHelper("PROMPT", prompt)
        self._printHelper("RAW PROMPT", raw_prompt)
        self._printHelper("RAW GENERATION", generation)

        generation = raw_prompt + generation

        matches = list(re.finditer(r"\bmodule\b", generation))
        if matches:
            last_match = matches[-1]
            generation = generation[last_match.start() :]

        # Balance the 'begin' and 'end' keywords
        lines = generation.split("\n")
        num_begins = 0
        num_ends = 0
        for line in lines:
            if "begin" in line and "//" not in line:
                num_begins += 1
            if "end" in line and "//" not in line:
                num_ends += 1

        end_additions = num_begins - num_ends
        if end_additions > 0:
            for i in range(end_additions):
                generation += "end\n"

        # Add the 'endmodule' keyword if it is not present
        if "endmodule" not in generation:
            generation += "\nendmodule\n"

        # Delete ```verilog from the generation
        if "```verilog" in generation:
            generation = generation.replace("```verilog", "").strip()

        self._printHelper("POST PROCESS GENERATION", generation)

        return {
            "task_id": task_id,
            "prompt": prompt,
            "test_path": self.get_path(task_id, suffix="tb"),
            "ref_path": self.get_path(task_id, suffix="answer"),
            "reasoning": reasoning,
            "generation": generation,
        }

    # Convert numpy arrays to lists for serialization
    def convert_ndarrays_to_lists(self, obj):
        if isinstance(obj, dict):
            return {
                key: self.convert_ndarrays_to_lists(value) for key, value in obj.items()
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def parse_ppa_json(self) -> list[dict]:
        # we read the problems from the PPA json list
        # to know in advance which problems require a call to openlane
        base_dir = Path(__file__).resolve().parent.parent
        golden_metrics_path = base_dir / "metrics/PPA_golden_solutions/verigen.json"
        if not golden_metrics_path.exists():
            raise FileNotFoundError(f"File not found: {golden_metrics_path}")
        with open(golden_metrics_path, "r") as fd:
            data: list[dict] = json.load(fd)
        return data

    def _evaluate_stx_fnc(self, generation: str, test_path: str, ref_path: str) -> dict:
        with tempfile.NamedTemporaryFile(
            suffix=".sv", delete=True, dir=self.path_temporary_files
        ) as f:
            f.write(generation.encode("utf-8"))
            f.flush()
            result = eval_script_verigen(
                Path(f.name), Path(test_path), Path(ref_path), self.simulator
            )
        return result

    def _evaluate_syn_ppa(
        self, result: dict, problem_id: str, generation_index: int, C: defaultdict
    ) -> dict:
        gen_dir = Path(
            os.path.join(
                self.base_gen_dir,
                "generated_problems",
                "answer_" + problem_id,
                f"generation_{generation_index+1}",
            )
        )
        openlane_result = run_openlane_for_generation(
            gen_dir,
            problem_id,
            "model_name",
        )

        if openlane_result[
            "metrics"
        ]:  # If metrics could be generated, synthesis passed
            result["synthesis_passed"] = True
            C["area"][problem_id].append(openlane_result["metrics"]["area"])
            C["power"][problem_id].append(openlane_result["metrics"]["power"])
            C["performance"][problem_id].append(
                openlane_result["metrics"]["performance"]
            )

        result.update(
            {
                "openlane_status": openlane_result["status"],
                "openlane_output": openlane_result.get("output", ""),
            }
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

    def _compute_ppa(self, ppa_data: list[dict], C: dict, n: int) -> dict:
        g = {"area": {}, "power": {}, "performance": {}}
        for entry in ppa_data:
            name = entry["prob_name"]
            short_name = name.replace(".json", "")
            g["area"][short_name] = entry["area"]
            g["power"][short_name] = entry["power"]
            g["performance"][short_name] = entry["performance"]

        # Auxiliar files to study extreme generation cases
        base_dir = os.path.dirname(self.metric_output_path)
        bad_ppa_path = os.path.join(base_dir, "bad_ppa.txt")
        better_than_human_path = os.path.join(base_dir, "better_than_human.txt")
        error_ppa_path = os.path.join(base_dir, "error_ppa.txt")

        for file_path in [bad_ppa_path, better_than_human_path, error_ppa_path]:
            with open(file_path, "w") as f:
                pass

        raw_power = compute_ppa_score(C["power"], g["power"], n, "power", base_dir)
        raw_performance = compute_ppa_score(
            C["performance"], g["performance"], n, "performance", base_dir
        )
        raw_area = compute_ppa_score(C["area"], g["area"], n, "area", base_dir)

        # Values are \in (0,2], where higher is worse
        # We want to flip the values and scale them to [0,100]
        def clip_and_flip_percentatge(val):
            return (2 - val) * 100

        return {
            "power": round(clip_and_flip_percentatge(raw_power), 2),
            "performance": round(clip_and_flip_percentatge(raw_performance), 2),
            "area": round(clip_and_flip_percentatge(raw_area), 2),
        }

    # evaluate the generations and report the results
    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        records = list()
        correct_syntax, correct_func, correct_synthesis, total = [], [], [], []

        # Dynamically read PPA from golden solutions
        ppa_data = self.parse_ppa_json()
        C = defaultdict(lambda: defaultdict(list))

        for i in range(len(references)):
            n_correct_syntax, n_correct_func, n_correct_synthesis = 0, 0, 0
            for j in range(len(generations[i])):  # N=5
                test_path = generations[i][j]["test_path"]
                ref_path = generations[i][j]["ref_path"]
                generation = generations[i][j]["generation"]

                problem_id = (
                    os.path.basename(ref_path).split("answer_")[-1].split(".")[0]
                )

                self._printHelper("TEST PATH", test_path)
                self._printHelper("REF PATH", ref_path)
                self._printHelper("GENERATION", generation)
                self._printHelper("PROBLEM ID", problem_id)

                result = self._evaluate_stx_fnc(generation, test_path, ref_path)
                if result["func_passed"]:
                    result = self._evaluate_syn_ppa(result, problem_id, j, C)

                n_correct_syntax += int(result["syntax_passed"])
                n_correct_func += int(result["func_passed"])
                n_correct_synthesis += int(result["synthesis_passed"])

                records.append(result)

            correct_syntax.append(n_correct_syntax)
            correct_func.append(n_correct_func)
            correct_synthesis.append(n_correct_synthesis)
            total.append(len(generations[i]))

        # Calculate pass@k (adapted from VerilogEval v1 repo)
        ks = [1, 5, 20]
        ret = {
            "syntax": self._compute_pass_k(
                np.array(correct_syntax), np.array(total), ks
            ),
            "func": self._compute_pass_k(np.array(correct_func), np.array(total), ks),
            "synthesis": self._compute_pass_k(
                np.array(correct_synthesis), np.array(total), ks
            ),
        }

        # Calculate PPA score
        ppa = self._compute_ppa(ppa_data, C, len(references))
        ret.update(ppa)

        return ret
