# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

"""
VerilogEval: Evaluating Large Language Models for Verilog Code Generation
https://arxiv.org/abs/2309.07544
VerlogEval benchmark implemented by HPAI team at Barcelona Supercomputing Center (BSC).
Homepage: https://github.com/NVlabs/verilog-eval
"""

import json
import os
import re
import tempfile
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk, load_dataset
from metrics.code_eval import estimate_pass_at_k
from metrics.code_eval import (
    compute_perplexities,
    compute_min_k,
    compute_min_k_veriContaminated,
)
from metrics.eval_verilog import eval_rtllm
from metrics.openlane_unified import (
    create_problem_structure,
    run_openlane_for_generation,
)
from metrics.ppa_score import compute_ppa_score
from src.turtle_eval.base import TaskExtension
from transformers import AutoTokenizer


os.environ["HF_ALLOW_CODE_EVAL"] = "1"

_CITATION = """
@inproceedings{liu2023verilogeval,
  title={{VerilogEval:} Evaluating Large Language Models for Verilog Code Generation},
  author={Liu, Mingjie and Pinckney, Nathaniel and Khailany, Brucek and Ren, Haoxing},
  booktitle={2023 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)}, 
  year={2023}
}
"""


class RTLLM(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        super().__init__(stop_words=[], requires_execution=False)
        kwargs = kwargs.get("kwargs", {})
        self.model = kwargs.get("model")
        self.tokenizer = kwargs.get("tokenizer")
        self.simulator = kwargs.get("simulator")

        self.path_temporary_files = kwargs.get("path_temporary_files")
        rtllm_version = kwargs.get("rtllm_version")
        prompt = kwargs.get("prompt", None)
        self.metric_output_path = kwargs.get("metric_output_path")
        self.prompt = kwargs.get("prompt", "")

        # Set-up basic params
        self.debug = True

        # Make sure we have access to the dataset and the repo
        path_rtllm_repo = os.path.join(os.path.dirname(__file__), "RTLLM")
        if not os.path.exists(path_rtllm_repo):
            raise ValueError(f"Path to `RTLLM` repo not found.\n`{path_rtllm_repo}` does not exists.")
        self.path_dataset = path_rtllm_repo
        self.generate_report = kwargs.get("generate_report", False)

        # Added for perplexity
        self.compute_ppl_only = kwargs.get("compute_ppl_only", False)
        self.kwargs = kwargs

        # Try to load dataset: HuggingFace Hub first, then local GPFS, then fail
        if rtllm_version == "1.1":
            hf_dataset_name = "ggcristian/RTLLM-v1.1"
            local_path = (
                "/chips-design/bigcode/datasets/RTLLM-v1.1"
            )
        elif rtllm_version == "2.0":
            hf_dataset_name = "ggcristian/RTLLM-v2.0"
            local_path = (
                "/chips-design/bigcode/datasets/RTLLM-v2.0"
            )
        else:
            raise ValueError("Please, provide rtllm_version flag.")

        # Try HuggingFace Hub first
        try:
            print(f"Attempting to load dataset from HuggingFace Hub: {hf_dataset_name}")
            self.dataset = load_dataset(hf_dataset_name, verification_mode="no_checks")
            print(f"Successfully loaded dataset from HuggingFace Hub")
        except Exception as e:
            print(f"Failed to load from HuggingFace Hub: {e}")
            # Try local path
            if os.path.exists(local_path):
                print(f"Loading dataset from local path: {local_path}")
                self.dataset = load_from_disk(local_path)
            else:
                raise FileNotFoundError(
                    f"Dataset not found on HuggingFace Hub ({hf_dataset_name}) or local path ({local_path})"
                )

        # setup directories for (later-on) computing the PPA
        self.load_generations_path = kwargs.get("load_generations_path", None)
        if self.load_generations_path is not None:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path)).resolve()
            _ = create_problem_structure(output_dir, json_path, mode="rtllm")
            self.base_gen_dir = output_dir
            self.debug = False

    # TODO: Delete this, just a helper function that will print contents generated for debug purposes
    def _printHelper(self, title: str, content: str):
        if self.debug:
            # fmt: off
            print("\n" + "-" * 30 + title + "-" * 30 + "\n" + content + "\n" + "-" * 70 + "\n")
            # fmt: on

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_file(self, path: str):
        with open(path) as fd:
            contents = fd.read()
        return contents

    def get_prompt(self, doc):
        full_prompt = self.get_file(
            os.path.join(self.path_dataset, doc["folder_path"], "design_description.txt")
        )

        # API models
        # we assume that closed source models apply already a chat template
        if self.tokenizer is None:
            return full_prompt.strip()

        # Local models
        # apply a chat template
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if tokenizer.chat_template is None:
            if self.prompt == "deepseek":
                tokenizer.chat_template = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
        conversation = [
            {"role": "user", "content": full_prompt.strip()},
        ]
        # Some models might not have a template,
        # in that case we return the raw prompt
        try:
            ret = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, thinking=True
            )
        except ValueError:
            print(f"Warning: {self.model} does not has a tokenizer template.")
            ret = full_prompt.strip()
        self._printHelper("PROMPT", ret)
        return ret

    def get_reference_path(self, doc):
        # as we did with VerilogEval impl we use regex
        folder_path = os.path.join(self.path_dataset, doc["folder_path"])
        regex = re.compile(r"^verified_.*\.v$")
        matching_files = [f for f in os.listdir(folder_path) if regex.match(f)]
        assert len(matching_files) == 1, (
            f"Golden solution not found or possible duplicate. Found files: {matching_files}"
        )
        file_path = os.path.join(folder_path, matching_files[0])
        return file_path

    def get_reference(self, doc):
        # as we did with VerilogEval impl we use regex
        return self.get_file(self.get_reference_path(doc))

    def postprocess_generation(self, generation, idx):
        # For reasoning models, we keep only the final answer
        if "assistantfinal" in generation:  # gpt-oss
            delimiter = "assistantfinal"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
        elif "</seed:think>" in generation:  # seed-oss-36b
            delimiter = "</seed:think>"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
        elif "</think>" in generation:
            delimiter = "</think>"
            reasoning, generation = generation.rsplit(delimiter, 1)
            reasoning = reasoning.strip()
        else:
            reasoning = None

        resp_content = generation.strip()

        # Start the process described here: https://github.com/NVlabs/verilog-eval/blob/508a4df32187ceb77945fe4a40b940e4b6dc3024/scripts/sv-generate#L425
        backticks_count = 0
        endmodule_before_startmodule = False
        module_already_exists = False

        for line in resp_content.splitlines():
            if line.startswith("```"):
                backticks_count += 1
            elif line.startswith("endmodule"):
                if not module_already_exists:
                    endmodule_before_startmodule = True
            elif line.startswith("module"):
                module_already_exists = True

        if endmodule_before_startmodule:
            module_already_exists = False

        # instead of using a file buffer (verilogeval original implementation)
        # let's use an in-memory data structure to avoid potential overhead
        output_lines = []

        # Second pass print out as appropriate
        found_first_backticks = False
        found_second_backticks = False
        found_module = False
        found_endmodule = False

        for line in resp_content.splitlines():
            echo_line = True

            if line.strip().startswith(
                "module"
            ):  # now we monitor if we've found it but we don't do anything with it.
                found_module = True

            if backticks_count >= 2:
                if (not found_first_backticks) or found_second_backticks:
                    echo_line = False
            else:
                if found_endmodule:
                    echo_line = False
                if module_already_exists and not found_module:
                    echo_line = False

            if line.startswith("```"):
                if not found_first_backticks:
                    found_first_backticks = True
                else:
                    found_second_backticks = True
                echo_line = False
            elif line.strip().startswith("endmodule"):
                found_endmodule = True

            if echo_line:
                if "RTLCoder" in self.model:
                    if line.strip().startswith("endmodule"):
                        line = line.rsplit("endmodule", 1)[0] + "\n" + "endmodule"
                output_lines.append(line)

        output_lines.append("")

        generation = "\n".join(output_lines)
        if "```verilog" in generation:
            generation = generation.replace(
                "```verilog", "```"
            )  # just to fallback to the next case to avoid duplicated logic
        if "```" in generation:
            # Regex taken from Langchain-rust repo: https://github.com/Abraxas-365/langchain-rust/blob/60c71258d8a40c278f2867f6b4d3871265f6e638/src/output_parsers/markdown_parser.rs#L13
            pattern = re.compile(r"```(?:\w+)?\s*([\s\S]+?)\s*```")
            match = pattern.search(generation)
            if match:
                generation = match.group(1)

        cwd = os.getcwd()
        test_path_abs = os.path.join(
            self.path_dataset, self.dataset["train"][idx]["folder_path"], "testbench.v"
        )
        ref_path_abs = self.get_reference_path(self.dataset["train"][idx])
        test_path_rel = os.path.relpath(test_path_abs, cwd)
        ref_path_rel = os.path.relpath(ref_path_abs, cwd)

        return {
            "prompt": self.get_prompt(self.dataset["train"][idx]),
            "test_path": test_path_rel,
            "ref_path": ref_path_rel,
            "reasoning": reasoning,
            "generation": generation,
        }

    def parse_ppa_json(self) -> list[dict]:
        # we read the problems from the PPA json list
        # to know in advance which problems require a call to openlane
        base_dir = Path(__file__).resolve().parent.parent
        golden_metrics_path = base_dir / "metrics/PPA_golden_solutions/rtllm.json"
        if not golden_metrics_path.exists():
            raise FileNotFoundError(f"File not found: {golden_metrics_path}")
        with open(golden_metrics_path, "r") as fd:
            data: list[dict] = json.load(fd)
        return data

    def _evaluate_stx_fnc(self, generation: str, test_path: str, ref_path: str) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".sv", delete=True, dir=self.path_temporary_files) as f:
            f.write(generation.encode("utf-8"))
            f.flush()
            result = eval_rtllm(
                Path(f.name), Path(test_path), Path(ref_path), simulator=self.simulator, flag=True
            )
        return result

    def _evaluate_syn_ppa(self, result: dict, problem_id: str, generation_index: int, C: defaultdict) -> dict:
        gen_dir = Path(
            os.path.join(
                self.base_gen_dir,
                "generated_problems",
                "verified_" + problem_id,
                f"generation_{generation_index + 1}",
            )
        )
        openlane_result = run_openlane_for_generation(
            gen_dir,
            problem_id,
            "model_name",
            pdk_root=None,
        )

        if openlane_result["status"] == "Success":  # If metrics could be generated, synthesis passed
            result["synthesis_passed"] = True
            C["area"][problem_id].append(openlane_result["metrics"]["area"])
            C["power"][problem_id].append(openlane_result["metrics"]["power"])
            C["performance"][problem_id].append(openlane_result["metrics"]["performance"])

        result.update(
            {
                "openlane_status": openlane_result["status"],
                "openlane_output": openlane_result.get("output", ""),
            }
        )
        return result, C

    def _compute_pass_k(self, correct: np.ndarray, total: np.ndarray, ks: list[int]) -> dict:
        results = {}
        for k in ks:
            if (total >= k).all():
                results[f"pass@{k}"] = round(estimate_pass_at_k(total, correct, k).mean() * 100, 2)
        return results

    def _compute_ppa(self, ppa_data: list[dict], C: dict, n: int) -> dict:
        g = {"area": {}, "power": {}, "performance": {}}
        for entry in ppa_data:
            name = entry["prob_name"]
            short_name = name.split("_", 1)[-1].replace(".json", "")
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
        raw_performance = compute_ppa_score(C["performance"], g["performance"], n, "performance", base_dir)
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
        reports = defaultdict(dict)
        correct_syntax, correct_func, correct_synthesis, total = [], [], [], []

        # Dynamically read PPA from golden solutions
        ppa_data = self.parse_ppa_json()
        ppa_ids = [
            d["prob_name"].split("_", 1)[-1].split(".")[0] for d in ppa_data
        ]  # 35_freq_by_odd.json => freq_by_odd
        C = defaultdict(lambda: defaultdict(list))

        for i in range(len(references)):
            n_correct_syntax, n_correct_func, n_correct_synthesis = 0, 0, 0
            for j in range(len(generations[i])):  # N=5
                test_path = generations[i][j]["test_path"]
                ref_path = generations[i][j]["ref_path"]
                generation = generations[i][j]["generation"]

                # handle both relative and absolute paths
                if "bigcode-evaluation-harness/" in test_path:
                    test_path = test_path.split("bigcode-evaluation-harness/", 1)[1]
                if "bigcode-evaluation-harness/" in ref_path:
                    ref_path = ref_path.split("bigcode-evaluation-harness/", 1)[1]

                test_path = os.path.join(os.getcwd(), test_path)
                ref_path = os.path.join(os.getcwd(), ref_path)

                problem_id = os.path.basename(ref_path).split("verified_")[-1].split(".")[0]

                result = self._evaluate_stx_fnc(generation, test_path, ref_path)
                if result["func_passed"] and problem_id in ppa_ids:
                    result, C = self._evaluate_syn_ppa(result, problem_id, j, C)

                n_correct_syntax += int(result["syntax_passed"])
                n_correct_func += int(result["func_passed"])
                n_correct_synthesis += int(result["synthesis_passed"])

                reports[problem_id][f"generation_{j + 1}"] = result["func_passed"]

            correct_syntax.append(n_correct_syntax)
            correct_func.append(n_correct_func)
            correct_synthesis.append(n_correct_synthesis)
            total.append(len(generations[i]))

        # Calculate pass@k (adapted from VerilogEval v1 repo)
        ks = [1, 5, 20]
        ret = {
            "syntax": self._compute_pass_k(np.array(correct_syntax), np.array(total), ks),
            "func": self._compute_pass_k(np.array(correct_func), np.array(total), ks),
            "synthesis": self._compute_pass_k(np.array(correct_synthesis), np.array(total), ks),
        }

        # Calculate PPA score
        ppa = self._compute_ppa(ppa_data, C, len(references))
        ret.update(ppa)

        if self.generate_report:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path)) / "report.json"
            with open(output_dir, "w") as f:
                json.dump(reports, f, indent=4)

        # Cleanup PPA generated problems dir
        if self.load_generations_path is not None:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path)) / "generated_problems"
            if output_dir.exists() and output_dir.is_dir():
                shutil.rmtree(output_dir)

        return ret
