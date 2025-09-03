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
from collections import defaultdict
from pathlib import Path

import numpy as np
from metrics.code_eval import estimate_pass_at_k
from metrics.eval_verilog import eval_verilog_eval
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


class VerilogEvalCodeComplete(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        super().__init__(stop_words=[], requires_execution=True)
        kwargs = kwargs.get("kwargs", {})
        self.model = kwargs.get("model")
        self.simulator = kwargs.get("simulator", "icarus")

        # Set-up basic params
        self.debug = False
        self.path_temporary_files = kwargs.get("path_temporary_files")
        self.task_name = "code-complete-iccad2023"
        self.prompt = kwargs.get("prompt", None)
        self.examples = kwargs.get("few_shot")
        assert self.examples >= 0 and self.examples <= 4, (
            "Few shot supported range is 0-4."
        )
        self.generate_report = kwargs.get("generate_report", False)

        # Make sure we have access to the dataset and the repo
        path_verilog_eval_repo = os.path.join(os.path.dirname(__file__), "verilog-eval")
        if not os.path.exists(path_verilog_eval_repo):
            raise ValueError(
                f"Path to `verilog_eval` repo not found.\n`{path_verilog_eval_repo}` does not exists."
            )

        self.path_dataset = os.path.join(
            path_verilog_eval_repo, "dataset_code-complete-iccad2023"
        )
        self.path_examples = os.path.join(path_verilog_eval_repo, "scripts")

        # setup directories for (later-on) computing the PPA
        self.load_generations_path = kwargs.get("load_generations_path", None)
        if self.load_generations_path is not None:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path) + "/")
            _ = create_problem_structure(output_dir, json_path)
            self.base_gen_dir = output_dir
            self.debug = False

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
        # VerilogEval has no official release of HF dataset, so we must build it locally from the repo
        ids = set()
        for file in os.listdir(self.path_dataset):
            if (
                not os.path.isdir(file)
                and file.startswith("Prob")
                and file.split(".")[-1] != "sv"
            ):
                # ProbXXX_<task_id>_<other>
                task_id = file.split("_", 1)[1].rsplit("_", 1)[0]
                ids.add(task_id)
        self.dataset = [{"task_id": id} for id in ids]  # mock HF dataset
        return self.dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        if self.examples == 0:
            return ""
        filename = f"verilog-example-prefix_{self.task_name}_{self.examples}-shot.txt"
        file = os.path.join(self.path_examples, filename)
        assert os.path.exists(file) == True, (
            f"Fewshot example for n_shot = {self.examples} not found or possible duplicate."
        )
        with open(file) as fd:
            contents = fd.read()
        return "\n" + contents

    def get_path(self, task_id: str, suffix: str):
        ext = "sv" if suffix in ["test", "ref"] else "txt"
        regex = re.compile(rf"^Prob\d+_{task_id}_{suffix}\.{ext}$")
        matching_files = [f for f in os.listdir(self.path_dataset) if regex.match(f)]
        assert len(matching_files) == 1, (
            f"Problem for task_id = {task_id} not found or possible duplicate. {matching_files}"
        )
        file = os.path.join(self.path_dataset, matching_files[0])
        return file

    def get_file(self, task_id: str, suffix: str):
        file = self.get_path(task_id, suffix)
        with open(file) as fd:
            contents = fd.read()
        return contents

    def get_prompt(self, doc):
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if tokenizer.chat_template is None:
            if self.prompt == "deepseek":
                tokenizer.chat_template = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
            elif self.prompt == "codellama":
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"

        system_msg = """
You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions.
"""
        prompt_prefix = """
// Implement the Verilog module based on the following description. Assume that signals are positive clock/clk triggered unless otherwise stated.
"""
        prefix = True
        prefixed_prompt = []
        prompt = self.get_file(doc["task_id"], suffix="prompt")
        for line in prompt.splitlines():
            if "module TopModule" in line:
                prefixed_prompt.append("")
                prefix = False
            if prefix:
                prefixed_prompt.append("// " + line)
            else:
                prefixed_prompt.append(line)
        prefixed_prompt = "\n".join(prefixed_prompt)
        full_prompt = ""
        full_prompt += system_msg
        full_prompt += self.fewshot_examples()
        full_prompt += prompt_prefix + prefixed_prompt

        # Note: the original implementation does not include the
        # system_msg as the LLM's template sys msg; so we do the same
        conversation = [
            {"role": "user", "content": full_prompt},
        ]

        try:
            ret = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            ret += "<think>\n"
        except ValueError:
            print(f"Warning: {self.model} does not have a tokenizer template.")
            ret = full_prompt

        return ret

    def get_reference(self, doc):
        return self.get_file(doc["task_id"], suffix="ref")

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
        if "assistantfinal" in generation:
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
        resp_content = generation

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
            elif line.startswith("module TopModule"):
                module_already_exists = True

        if endmodule_before_startmodule:
            module_already_exists = False

        # instead of using a file buffer (verilogeval original implementation)
        # let's use an in-memory data structure to avoid potential overhead
        output_lines = []

        # if module doesn't exist (which it shouldn't for code completition) then print out interface
        if not module_already_exists:
            ifc_content = self.get_file(task_id, suffix="ifc")
            output_lines.append(ifc_content.strip())
            output_lines.append("")  # this will get parsed as a "\n"

        # Second pass print out as appropriate
        found_first_backticks = False
        found_second_backticks = False
        found_module = False
        found_endmodule = False

        for line in resp_content.splitlines():
            echo_line = True

            if line.strip().startswith(
                "module TopModule"
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

        if backticks_count == 1 or backticks_count > 2:
            output_lines.append("// VERILOG-EVAL: abnormal backticks count")
            output_lines.append("")
        if found_module:
            output_lines.append(
                "// VERILOG-EVAL: errant inclusion of module definition"
            )
            output_lines.append("")
        if not found_endmodule:
            output_lines.append("// VERILOG-EVAL: endmodule not found")
            output_lines.append("")

        output_lines.append("")

        return {
            "prompt": self.get_prompt(self.dataset[idx]),
            "test_path": self.get_path(task_id, suffix="test"),
            "ref_path": self.get_path(task_id, suffix="ref"),
            "reasoning": reasoning,
            "generation": "\n".join(output_lines),
        }

    def parse_ppa_json(self) -> list[dict]:
        # we read the problems from the PPA json list
        # to know in advance which problems require a call to openlane
        base_dir = Path(__file__).resolve().parent.parent
        golden_metrics_path = base_dir / "metrics/PPA_golden_solutions/verilogeval.json"
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
            result = eval_verilog_eval(
                Path(f.name), Path(test_path), Path(ref_path), self.simulator
            )
        return result

    def _evaluate_syn_ppa(
        self, result: dict, problem_id: str, generation_index: int, C: defaultdict
    ) -> tuple:
        gen_dir = Path(
            os.path.join(
                self.base_gen_dir,
                "generated_problems",
                problem_id,
                f"generation_{generation_index + 1}",
            )
        )
        openlane_result = run_openlane_for_generation(
            gen_dir,
            problem_id,
            "model_name",
        )

        if (
            openlane_result["status"] == "Success"
        ):  # If metrics could be generated, synthesis passed
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
        return result, C

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
            g["area"][name] = entry["area"]
            g["power"][name] = entry["power"]
            g["performance"][name] = entry["performance"]

        raw_power = compute_ppa_score(C["power"], g["power"], n, "power")
        raw_performance = compute_ppa_score(
            C["performance"], g["performance"], n, "performance"
        )
        raw_area = compute_ppa_score(C["area"], g["area"], n, "area")

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
        reports = defaultdict(list)
        correct_syntax, correct_func, correct_synthesis, total = [], [], [], []

        # Dynamically read PPA from golden solutions
        ppa_data = self.parse_ppa_json()
        ppa_ids = [d["prob_name"] for d in ppa_data]  # ProbXXX
        C = defaultdict(lambda: defaultdict(list))

        for i in range(len(references)):
            n_correct_syntax, n_correct_func, n_correct_synthesis = 0, 0, 0
            for j in range(len(generations[i])):  # N=5
                test_path = generations[i][j]["test_path"]
                ref_path = generations[i][j]["ref_path"]
                generation = generations[i][j]["generation"]

                problem_id = os.path.basename(ref_path).split("_")[0]

                result = self._evaluate_stx_fnc(generation, test_path, ref_path)
                if result["func_passed"] and problem_id in ppa_ids:
                    result, C = self._evaluate_syn_ppa(result, problem_id, j, C)

                n_correct_syntax += int(result["syntax_passed"])
                n_correct_func += int(result["func_passed"])
                n_correct_synthesis += int(result["synthesis_passed"])

                reports[problem_id].append(
                    {
                        f"generation_{j + 1}": result["func_passed"],
                        "error": (
                            result["passfail"] if not result["func_passed"] else None
                        ),
                    }
                )

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


class VerilogEvalRTLToSpecification(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        super().__init__(stop_words=[], requires_execution=True)
        kwargs = kwargs.get("kwargs", {})
        self.model = kwargs.get("model")
        self.simulator = kwargs.get("simulator", "icarus")

        # Set-up basic params
        self.debug = False
        self.path_temporary_files = kwargs.get("path_temporary_files")
        self.task_name = "spec-to-rtl"
        self.prompt = kwargs.get("prompt", None)
        self.examples = kwargs.get("few_shot", 0)
        assert 0 <= self.examples <= 4, "Few shot supported range is 0-4."
        self.generate_report = kwargs.get("generate_report", False)

        # Make sure we have access to the dataset and the repo
        path_verilog_eval_repo = os.path.join(os.path.dirname(__file__), "verilog-eval")
        if not os.path.exists(path_verilog_eval_repo):
            raise ValueError(
                f"Path to `verilog_eval` repo not found.\n`{path_verilog_eval_repo}` does not exist."
            )
        self.path_dataset = os.path.join(path_verilog_eval_repo, "dataset_spec-to-rtl")
        self.path_examples = os.path.join(path_verilog_eval_repo, "scripts")

        # setup directories for (later-on) computing the PPA
        self.load_generations_path = kwargs.get("load_generations_path", None)
        if self.load_generations_path is not None:
            json_path = self.load_generations_path
            output_dir = Path(os.path.dirname(json_path) + "/")
            _ = create_problem_structure(output_dir, json_path)
            self.base_gen_dir = output_dir
            self.debug = False

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
        # VerilogEval has no official release of HF dataset, so we must build it locally from the repo
        ids = set()
        for file in os.listdir(self.path_dataset):
            if (
                not os.path.isdir(file)
                and file.startswith("Prob")
                and file.split(".")[-1] != "sv"
            ):
                # ProbXXX_<task_id>_<other>
                task_id = file.split("_", 1)[1].rsplit("_", 1)[0]
                ids.add(task_id)
        self.dataset = [{"task_id": id} for id in ids]  # mock HF dataset
        return self.dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        if self.examples == 0:
            return ""
        filename = f"verilog-example-prefix_{self.task_name}_{self.examples}-shot.txt"
        file = os.path.join(self.path_examples, filename)
        assert os.path.exists(file) == True, (
            f"Fewshot example for n_shot = {self.examples} not found or possible duplicate."
        )
        with open(file) as fd:
            contents = fd.read()
        return "\n" + contents

    def get_path(self, task_id: str, suffix: str):
        ext = "sv" if suffix in ["test", "ref"] else "txt"
        regex = re.compile(rf"^Prob\d+_{task_id}_{suffix}\.{ext}$")
        matching_files = [f for f in os.listdir(self.path_dataset) if regex.match(f)]
        assert len(matching_files) == 1, (
            f"Problem for task_id = {task_id} not found or possible duplicate. {matching_files}"
        )
        file = os.path.join(self.path_dataset, matching_files[0])
        return file

    def get_file(self, task_id: str, suffix: str):
        file = self.get_path(task_id, suffix)
        with open(file) as fd:
            contents = fd.read()
        return contents

    def get_prompt(self, doc):
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if tokenizer.chat_template is None:
            if self.prompt == "deepseek":
                tokenizer.chat_template = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
            elif self.prompt == "codellama":
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"

        system_msg = """
You are a Verilog RTL designer that only writes code using correct Verilog syntax.
"""
        prompt_no_explain_suffix = """
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet and do NOT output anything else.
"""
        prompt = self.get_file(doc["task_id"], suffix="prompt")
        full_prompt = ""
        full_prompt += system_msg
        full_prompt += self.fewshot_examples()
        full_prompt += "\nQuestion:\n"
        full_prompt += prompt.strip() + "\n"
        full_prompt = full_prompt.rstrip() + "\n" + prompt_no_explain_suffix
        full_prompt += "\nAnswer:\n"

        # Note: the original implementation does not include the
        # system_msg as the LLM's template sys msg; so we do the same
        conversation = [
            {"role": "user", "content": full_prompt},
        ]

        try:
            ret = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            ret += "<think>\n"
        except ValueError:
            print(f"Warning: {self.model} does not have a tokenizer template.")
            ret = full_prompt

        self._printHelper("PROMPT", ret)
        return ret

    def get_reference(self, doc):
        return self.get_file(doc["task_id"], suffix="ref")

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
        if "assistantfinal" in generation:
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
        resp_content = generation

        # Start the process described here: https://github.com/NVlabs/verilog-eval/blob/508a4df32187ceb77945fe4a40b940e4b6dc3024/scripts/sv-generate#L503
        backticks_count = 0
        endmodule_before_startmodule = False
        module_already_exists = False

        for line in resp_content.splitlines():
            if line.startswith("```"):
                backticks_count += 1
            elif line.startswith("endmodule"):
                if not module_already_exists:
                    endmodule_before_startmodule = True
            elif line.startswith("module TopModule"):
                module_already_exists = True

        if endmodule_before_startmodule:
            module_already_exists = False

        # instead of using a file buffer (verilogeval original implementation)
        # let's use an in-memory data structure to avoid potential overhead
        output_lines = []

        found_code_lines = []
        found_code_start = False
        found_code_end = False
        for line in resp_content.splitlines():
            if not found_code_start:
                if line.strip() == "[BEGIN]":
                    found_code_start = True
                elif line.lstrip().startswith("[BEGIN]"):
                    found_code_lines.append(line.lstrip().replace("[BEGIN]", ""))
                    found_code_start = True
            elif found_code_start and not found_code_end:
                if line.strip() == "[DONE]":
                    found_code_end = True
                elif line.rstrip().endswith("[DONE]"):
                    found_code_lines.append(line.rstrip().replace("[DONE]", ""))
                    found_code_end = True
                elif (
                    "[DONE]" in line
                ):  # For Llama 3.1 8B, [DONE] is followed by comments
                    found_code_end = True
                else:
                    found_code_lines.append(line)

        if found_code_start and found_code_end:
            for line in found_code_lines:
                output_lines.append(line)

        if not found_code_start and not found_code_end:
            # fallback to code completion style extraction

            # Second pass print out as appropriate
            found_first_backticks = False
            found_second_backticks = False
            found_module = False
            found_endmodule = False

            for line in resp_content.splitlines():
                echo_line = True

                if line.strip().startswith(
                    "module TopModule"
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
        generation = generation.replace("```verilog", "```")
        if "```" in generation:
            # Regex taken from Langchain-rust repo: https://github.com/Abraxas-365/langchain-rust/blob/60c71258d8a40c278f2867f6b4d3871265f6e638/src/output_parsers/markdown_parser.rs#L13
            pattern = re.compile(r"```(?:\w+)?\s*([\s\S]+?)\s*```")
            match = pattern.search(generation)
            if match:
                generation = match.group(1)

        return {
            "prompt": self.get_prompt(self.dataset[idx]),
            "test_path": self.get_path(task_id=task_id, suffix="test"),
            "ref_path": self.get_path(task_id=task_id, suffix="ref"),
            "reasoning": reasoning,
            "generation": generation,
        }

    def parse_ppa_json(self) -> list[dict]:
        # we read the problems from the PPA json list
        # to know in advance which problems require a call to openlane
        base_dir = Path(__file__).resolve().parent.parent
        golden_metrics_path = base_dir / "metrics/PPA_golden_solutions/verilogeval.json"
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
            result = eval_verilog_eval(
                Path(f.name), Path(test_path), Path(ref_path), self.simulator
            )
        return result

    def _evaluate_syn_ppa(
        self, result: dict, problem_id: str, generation_index: int, C: defaultdict
    ) -> tuple:
        gen_dir = Path(
            os.path.join(
                self.base_gen_dir,
                "generated_problems",
                problem_id,
                f"generation_{generation_index + 1}",
            )
        )
        openlane_result = run_openlane_for_generation(
            gen_dir,
            problem_id,
            "model_name",
        )

        if (
            openlane_result["status"] == "Success"
        ):  # If metrics could be generated, synthesis passed
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
        return result, C

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
            g["area"][name] = entry["area"]
            g["power"][name] = entry["power"]
            g["performance"][name] = entry["performance"]

        raw_power = compute_ppa_score(C["power"], g["power"], n, "power")
        raw_performance = compute_ppa_score(
            C["performance"], g["performance"], n, "performance"
        )
        raw_area = compute_ppa_score(C["area"], g["area"], n, "area")

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
        reports = defaultdict(list)
        correct_syntax, correct_func, correct_synthesis, total = [], [], [], []

        # Dynamically read PPA from golden solutions
        ppa_data = self.parse_ppa_json()
        ppa_ids = [d["prob_name"] for d in ppa_data]  # ProbXXX
        C = defaultdict(lambda: defaultdict(list))

        for i in range(len(references)):
            n_correct_syntax, n_correct_func, n_correct_synthesis = 0, 0, 0
            for j in range(len(generations[i])):  # N=5
                test_path = generations[i][j]["test_path"]
                ref_path = generations[i][j]["ref_path"]
                generation = generations[i][j]["generation"]

                problem_id = os.path.basename(ref_path).split("_")[0]

                result = self._evaluate_stx_fnc(generation, test_path, ref_path)
                if result["func_passed"] and problem_id in ppa_ids:
                    result, C = self._evaluate_syn_ppa(result, problem_id, j, C)

                n_correct_syntax += int(result["syntax_passed"])
                n_correct_func += int(result["func_passed"])
                n_correct_synthesis += int(result["synthesis_passed"])

                reports[problem_id].append(
                    {
                        f"generation_{j + 1}": result["func_passed"],
                        "error": (
                            result["passfail"] if not result["func_passed"] else None
                        ),
                    }
                )

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
