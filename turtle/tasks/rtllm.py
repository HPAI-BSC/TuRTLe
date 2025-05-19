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
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from metrics.code_eval import estimate_pass_at_k
from metrics.eval_verilog import eval_rtllm
from metrics.openlane_unified import (create_problem_structure,
                                      run_openlane_for_generation)
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
        self.path_temporary_files = kwargs.get("path_temporary_files")
        self.path_dataset_test = kwargs.get("path_dataset_test")
        prompt = kwargs.get("prompt", None)
        self.prompt = prompt if prompt is None else None

        # Set-up basic params
        self.debug = True

        # Make sure we have access to the dataset and the repo
        self.path_rtllm_repo = os.path.join(os.path.dirname(__file__), "RTLLM")
        if not os.path.exists(self.path_rtllm_repo):
            raise ValueError(
                f"Path to `RTLLM` repo not found.\n`{self.path_rtllm_repo}` does not exists."
            )
        
        self.dataset = load_from_disk(self.path_dataset_test)

        # setup directories for (later-on) computing the PPA
        self.load_generations_path = kwargs.get("load_generations_path", None)
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
            print("\n" + "-" * 30 + title + "-" * 30 + "\n" + content + "\n" + "-" * 70 + "\n")

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
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if tokenizer.chat_template is None:
            if self.prompt == "deepseek":
                tokenizer.chat_template = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
        full_prompt = self.get_file(os.path.join(self.path_rtllm_repo, doc["folder_path"], "design_description.txt"))
        conversation = [ {"role": "user", "content": full_prompt.strip()}, ]
        # Some models might not have a template,
        # in that case we return the raw prompt
        try:
            ret = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            print(f"Warning: {self.model} does not has a tokenizer template.")
            ret = full_prompt.strip()
        return ret

    def get_reference_path(self, doc):
        # as we did with VerilogEval impl we use regex
        folder_path = os.path.join(self.path_rtllm_repo, doc["folder_path"])
        regex = re.compile(r"^verified_.*\.v$")
        matching_files = [f for f in os.listdir(folder_path) if regex.match(f)]
        assert (
            len(matching_files) == 1
        ), f"Golden solution not found or possible duplicate. Found files: {matching_files}"
        file_path = os.path.join(folder_path, matching_files[0])
        return file_path

    def get_reference(self, doc):
        # as we did with VerilogEval impl we use regex
        return self.get_file(self.get_reference_path(doc))

    def postprocess_generation(self, generation, idx):

        # For reasoning models, we keep only the final answer
        if "</think>" in generation:
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
            # just to fallback to the next case to avoid duplicated logic
            generation = generation.replace("```verilog", "```")

        if "```" in generation:
            # Regex taken from Langchain-rust repo: https://github.com/Abraxas-365/langchain-rust/blob/60c71258d8a40c278f2867f6b4d3871265f6e638/src/output_parsers/markdown_parser.rs#L13
            pattern = re.compile(r"```(?:\w+)?\s*([\s\S]+?)\s*```")
            match = pattern.search(generation)
            if match:
                generation = match.group(1)

        return {
            "prompt": self.get_prompt(self.dataset["train"][idx]),
            "test_path": os.path.join(
                self.path_rtllm_repo,
                self.dataset["train"][idx]["folder_path"],
                "testbench.v",
            ),
            "ref_path": self.get_reference_path(self.dataset["train"][idx]),
            "reasoning": reasoning,
            "generation": generation,
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
        records = list()
        correct_syntax, correct_func, correct_synthesis, total = [], [], [], []
        C = {"area": {}, "power": {}, "performance": {}}  # to compute PPA score
        for i in range(len(references)):
            n_correct_syntax = 0
            n_correct_func = 0
            n_correct_synthesis = 0
            for j in range( len(generations[i])):  # can be either 1 or 20 samples in our case
                test_path = generations[i][j]["test_path"]
                ref_path = generations[i][j]["ref_path"]
                generation = generations[i][j]["generation"]
                problem_id = (os.path.basename(ref_path).split("verified_")[-1].split(".")[0])
                # Evaluate STX & FNC through Icarus Verilog
                with tempfile.NamedTemporaryFile(suffix=".sv", delete=True, dir=self.path_temporary_files) as f:
                    f.write(generation.encode("utf-8"))
                    f.flush()
                    result = eval_rtllm(
                        Path(f.name), Path(test_path), Path(ref_path)
                    )
                result["synthesis_passed"] = False
                if result["func_passed"] == True:
                    # Evaluate SYN & PPA through OpenLane
                    gen_dir = Path(os.path.join(self.base_gen_dir, "generated_problems", "verified_" + problem_id, f"generation_{j+1}"))
                    openlane_result = run_openlane_for_generation(gen_dir, problem_id, "model_name")
                    if openlane_result["metrics"]:  # If metrics could be generated, synthesis passed
                        result["synthesis_passed"] = True
                        # Check if the list exists, if not, create it
                        if problem_id not in C["area"]:
                            C["area"][problem_id] = []
                            C["power"][problem_id] = []
                            C["performance"][problem_id] = []

                        # Metrics of the LLM's generation
                        C["area"][problem_id].append(openlane_result["metrics"]["area"])
                        C["power"][problem_id].append(openlane_result["metrics"]["power"])
                        C["performance"][problem_id].append(openlane_result["metrics"]["performance"])

                    result.update(
                        {
                            "openlane_status": openlane_result["status"],
                            "openlane_output": openlane_result.get("output", ""),
                        }
                    )
                records.append(result)
                n_correct_syntax += int(result["syntax_passed"])
                n_correct_func += int(result["func_passed"])
                n_correct_synthesis += int(result["synthesis_passed"])
            correct_syntax.append(n_correct_syntax)
            correct_func.append(n_correct_func)
            correct_synthesis.append(n_correct_synthesis)
            total.append(len(generations[i]))

        # Calculate pass@k (taken from VerilogEval v1 repo)
        ret = {"syntax": None, "func": None}
        total = np.array(total)
        ks = [1, 5, 20]

        # pass@k STX
        correct_syntax = np.array(correct_syntax)
        ret["syntax"] = { f"pass@{k}": estimate_pass_at_k(total, correct_syntax, k).mean() for k in ks if (total >= k).all() }

        # pass@k FNC
        correct_func = np.array(correct_func)
        ret["func"] = { f"pass@{k}": estimate_pass_at_k(total, correct_func, k).mean() for k in ks if (total >= k).all() }

        # pass@k SYN
        correct_synthesis = np.array(correct_synthesis)
        ret["synthesis"] = { f"pass@{k}": estimate_pass_at_k(total, correct_synthesis, k).mean() for k in ks if (total >= k).all() }

        # Compute PPA score
        # Dynamically read PPA from golden solutions
        base_dir = Path(__file__).resolve().parent.parent
        golden_metrics_path = base_dir / "metrics/PPA_golden_solutions/rtllm.json"
        if not golden_metrics_path.exists():
            raise FileNotFoundError(f"File not found: {golden_metrics_path}")
        with open(golden_metrics_path, "r") as file:
            data = json.load(file)
        g = {"area": {}, "power": {}, "performance": {}}
        for entry in data:
            name = entry["prob_name"]
            short_name = name.split("_", 1)[-1].replace(".json", "")
            g["area"][short_name] = entry["area"]
            g["power"][short_name] = entry["power"]
            g["performance"][short_name] = entry["performance"]

        raw_power = compute_ppa_score(C["power"], g["power"], len(references), "power")
        raw_performance = compute_ppa_score(
            C["performance"], g["performance"], len(references), "performance"
        )
        raw_area = compute_ppa_score(C["area"], g["area"], len(references), "area")

        # Values are \in (0,2], where higher is worse
        # We want to flip the values and scale them to [0,100]
        def clip_and_flip_percentatge(val):
            return (1 - (val / 2)) * 100

        ret["area"] = round(clip_and_flip_percentatge(raw_area), 2)
        ret["power"] = round(clip_and_flip_percentatge(raw_power), 2)
        ret["performance"] = round(clip_and_flip_percentatge(raw_performance), 2)

        return ret
