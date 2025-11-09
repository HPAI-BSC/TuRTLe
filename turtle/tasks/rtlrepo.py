"""
RTL-Repo: A Benchmark for Evaluating LLMs on Large-Scale RTL Design Projects
https://arxiv.org/pdf/2405.17378

The RTL-Repo benchmark provides a valuable resource for the hardware design
community to assess and compare LLMs’ performance in realworld RTL design scenarios
and train LLMs specifically for Verilog code generation in complex, multi-file RTL projects.

Homepage: https://github.com/AUCOHL/RTL-Repo
"""

import csv
import os
import re
import math

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy DataLoader for API mode
    class DataLoader:
        pass

from datasets import Dataset, concatenate_datasets, load_from_disk
from metrics.code_eval import estimate_pass_at_k
from src.turtle_eval.base import TaskExtension
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_CITATION = """
@misc{allam2024rtlrepobenchmarkevaluatingllms,
      title={RTL-Repo: A Benchmark for Evaluating LLMs on Large-Scale RTL Design Projects}, 
      author={Ahmed Allam and Mohamed Shalan},
      year={2024},
      eprint={2405.17378},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.17378}, 
}
"""


class RTLRepo(TaskExtension):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, **kwargs):
        from vllm import LLM, SamplingParams

        # model, rtlrepo_use_train_ds, max_length_generation, path_data_benchmark, path_dataset_test, path_temporary_files):
        super().__init__(
            stop_words=[],
            requires_execution=False,
        )

        kwargs = kwargs.get("kwargs", {})
        self.kwargs = kwargs

        self.path_dataset_test = kwargs.get("path_dataset_test")
        self.rtlrepo_use_train_ds = kwargs.get("rtlrepo_use_train_ds")
        self.model = kwargs.get("model")
        self.max_length_generation = kwargs.get("max_length_generation")
        self.metric_output_path = kwargs.get("metric_output_path")

        print(f"Loading RTL-Repo dataset from {self.path_dataset_test}")
        print(f"Evaluating model {self.model}")
        print(f"Using train+test dataset? {self.rtlrepo_use_train_ds}")
        print(f"Max length generation: {self.max_length_generation}")

        self.max_length_prompt = self.max_length_generation - 80

        ds = load_from_disk(self.path_dataset_test)
        if self.rtlrepo_use_train_ds:
            self.dataset = concatenate_datasets([ds["train"], ds["test"]])
        else:
            self.dataset = ds["test"]

        # Use the same tokenizer as the model being evaluated
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        cross_file_prompt = f"// Repo Name: {doc['repo_name']}\n"

        for snippet in doc["context"]:
            cross_file_prompt += (
                f"// Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
            )

        in_file_prompt = f"// Path: {doc['file_path']}\n{doc['cropped_code']}\n"

        if self.tokenizer is not None and self.max_length_prompt is not None:
            cross_file_prompt_token_nums = len(self.tokenizer.encode(cross_file_prompt))
            in_file_prompt_token_nums = len(self.tokenizer.encode(in_file_prompt))

            exceed_token_nums = (
                cross_file_prompt_token_nums
                + in_file_prompt_token_nums
                - self.max_length_prompt
            )

            if exceed_token_nums > 0:
                cross_file_prompt_lines = cross_file_prompt.split("\n")
                for i in range(len(cross_file_prompt_lines) - 1, -1, -1):
                    exceed_token_nums -= len(
                        self.tokenizer.encode(cross_file_prompt_lines[i])
                    )

                    if exceed_token_nums < 0:
                        break

                cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"

        prompt = cross_file_prompt + in_file_prompt
        prompt = re.sub(r"\n{4,}", "\n\n", prompt)

        # Truncate the prompt so that its tokenized version respects self.max_length_prompt
        # This procedure of the original benchmark fails as only truncates cross_file_prompt
        # which is not enough for samples with large in_file_prompt.
        # This is a problem, since truncating in_file_prompt results in a prompt that is not valid
        # to predict the next line. That's why we truncate the prompt from the beggining.
        tokenized = self.tokenizer(prompt, truncation=False)["input_ids"]
        truncated = tokenized[-self.max_length_prompt :]  # truncate from the beggining
        prompt = self.tokenizer.decode(truncated, skip_special_tokens=True)

        if self.model == "Qwen3-235B-A22B":  # disable reasoning
            message = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["next_line"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        # Take the first non-empty, non-comment line
        # If all lines are comments, take the last comment line
        lines = generation.split("\n")
        last_comment_line = ""

        for line in lines:
            if line.strip():
                if not line.strip().startswith("//"):
                    return line
                last_comment_line = line

        return last_comment_line

    def _compute_perplexities(self, references):
        tensor_parallel_size = self.kwargs.get("ppl_tensor_parallel_size", 4)
        gpu_memory_utilization = self.kwargs.get("ppl_gpu_memory_utilization", 0.5)
        max_model_len = self.kwargs.get("ppl_max_model_len", 8192)
        dtype = self.kwargs.get("ppl_dtype", "bfloat16")
        swap_space = self.kwargs.get("ppl_swap_space", 32)

        print(f"Loading model: {self.model}")
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            dtype=dtype,
            swap_space=swap_space,
        )

        vllm_tok = self.llm.get_tokenizer()
        perplexities = []

        ds = self.get_dataset()
        for i, doc in tqdm(enumerate(ds)):
            prompt = self.get_prompt(doc)
            ref = references[i]

            full_text = prompt + ref

            # Tokenize prompt and full text to identify response token positions
            response_tokens = vllm_tok.encode(ref, add_special_tokens=False)
            full_tokens = vllm_tok.encode(full_text, add_special_tokens=False)

            # ensure max_model_len is respected
            if len(full_tokens) > max_model_len:
                print(
                    f"WARNING: prompt+response exceeds max_model_len. Truncating from the left..."
                )
                # Truncate from the left to maintain the response at the end
                full_tokens = full_tokens[-max_model_len:]

            # print(f"len(response_tokens): {len(response_tokens)}")
            # print(f"len(full_tokens): {len(full_tokens)}")

            # Find where response tokens start in the full text
            response_start_idx = len(full_tokens) - len(response_tokens)

            # Use vLLM to get log probabilities for the truncated full text
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,  # We only need the logprobs, not generation
                prompt_logprobs=1,  # Outputs top k most probable tokens + prompt token if not in the top
            )

            outputs = self.llm.generate(
                {"prompt_token_ids": full_tokens}, sampling_params
            )
            output = outputs[0]  # There is only one output
            # print(output.prompt_logprobs) # list of #full_tokens elements, each element is a dict with key "token_id" and value "Logprob"
            # Value example: Logprob(logprob=-3.3348352909088135, rank=3, decoded_token='110')
            # print(f"response_tokens: {response_tokens}")
            # print(f"output.prompt_logprobs[response_start_idx:]: {output.prompt_logprobs[response_start_idx:]}")

            # Extract log probabilities only for response tokens
            response_log_probs = []
            response_logprobs = output.prompt_logprobs[
                response_start_idx:
            ]  # each element is a dict with key "token_id" and value "Logprob"
            # Update response_tokens, because the concatenation of prompt+response could result in a slightly different tokenization
            response_tokens = full_tokens[response_start_idx:]

            # Skip prompt tokens and collect response token log probs
            for i, token_logprobs in enumerate(response_logprobs):
                # print(f"Token idx {i}, actual token id: {response_tokens[i]}, token_logprobs: {token_logprobs}")
                logprob_value = token_logprobs[response_tokens[i]].logprob
                response_log_probs.append(logprob_value)

            # Calculate perplexity: exp(-sum(log_probs) / num_tokens)
            avg_log_prob = sum(response_log_probs) / len(response_log_probs)
            perplexity = math.exp(-avg_log_prob)
            perplexities.append(perplexity)

            # Proactively release any cached VRAM between requests
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        valid_perplexities = [p for p in perplexities if not math.isinf(p) and p < 1000]

        return {
            "num_samples": len(references),
            "num_valid_samples": len(valid_perplexities),
            "perplexities": perplexities,
            "statistics": {
                "mean_perplexity": np.mean(valid_perplexities),
                "median_perplexity": np.median(valid_perplexities),
                "std_perplexity": np.std(valid_perplexities),
                "min_perplexity": np.min(valid_perplexities),
                "max_perplexity": np.max(valid_perplexities),
            },
        }

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        :return: dict[str: float]
        """
        # GitHub code for artifact evaluation
        # generations could be a list of lists if n_samples > 1
        if isinstance(generations[0], list):  # Check if it's a list of lists
            n_samples = len(generations[0])
            total_samples = len(generations) * n_samples
        else:
            total_samples = len(generations)
            n_samples = 1

        #################
        # Calculate pass@k for exact_match
        #################
        if n_samples > 1:
            correct_em = []
            for i in range(len(references)):
                n_correct_em = 0
                for j in range(
                    len(generations[i])
                ):  # can be either 1 or 20 samples in our case
                    generation = generations[i][j]
                    result = 1 if generation.split() == references[i].split() else 0
                    n_correct_em += result
                correct_em.append(n_correct_em)

            ks = [1, 5]
            total = np.full(len(correct_em), n_samples)
            correct_em = np.array(correct_em)
            pass_em = {
                f"pass@{k}": estimate_pass_at_k(total, correct_em, k).mean()
                for k in ks
                if (total >= k).all()
            }
        else:  # n_samples == 1
            exact_match = 0
            for pred, gt in zip(generations, references):
                if pred[0].split() == gt.split():
                    exact_match += 1
            pass_em = round(exact_match / total_samples, 5)

        return {"exact_match": pass_em}

        '''# generations could be a list of lists if n_samples > 1
        if isinstance(generations[0], list):  # Check if it's a list of lists
            n_samples = len(generations[0])
            total_samples = len(generations) * n_samples
        else:
            total_samples = len(generations)
            n_samples = 1

        #################
        # Calculate average for edit similarity
        # NOT used in the final evaluation
        #################
        """
        from fuzzywuzzy import fuzz
        edit_sim = 0.0
        if n_samples > 1:
            for idx, gt in enumerate(references):
                for sample in generations[idx]: # n_samples
                    edit_sim += fuzz.ratio(sample, gt)
        else: # n_samples == 1
            for pred, gt in zip(generations, references):
                edit_sim += fuzz.ratio(pred[0], gt)

        mean_es = round(edit_sim / total_samples, 5)"""

        #################
        # Calculate perplexity
        #################
        perplexity_results = self._compute_perplexities(references)
        perplexities = perplexity_results['perplexities']

        #################
        # Calculate pass@k for exact_match
        #################
        if n_samples > 1:
            correct_em = []
            for i in range(len(references)):
                n_correct_em = 0
                for j in range(
                    len(generations[i])
                ):  # can be either 1 or 20 samples in our case
                    generation = generations[i][j]
                    result = 1 if generation.split() == references[i].split() else 0
                    n_correct_em += result
                print(
                    f"Sample {i}: n_correct_em = {n_correct_em}, perplexity = {perplexities[i]}"
                )
                correct_em.append(n_correct_em)

            ks = [1, 5]
            total = np.full(len(correct_em), n_samples)
            correct_em = np.array(correct_em)
            pass_em = {
                f"pass@{k}": estimate_pass_at_k(total, correct_em, k).mean()
                for k in ks
                if (total >= k).all()
            }

            # Write csv for Perplexity vs Exact Match study
            results_dir = os.path.dirname(self.metric_output_path)
            with open(
                os.path.join(results_dir, "em_vs_ppl.csv"), "w", newline=""
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["correct_em", "perplexity"])
                for em, ppl in zip(correct_em, perplexities):
                    writer.writerow([em, ppl])

        else:  # n_samples == 1
            exact_match = 0
            for pred, gt in zip(generations, references):
                if pred[0].split() == gt.split():
                    exact_match += 1
            pass_em = round(exact_match / total_samples, 5)

        return {"exact_match": pass_em,
                "mean_perplexity": perplexity_results['statistics']['mean_perplexity'],
                "median_perplexity": perplexity_results['statistics']['median_perplexity'],
                "num_samples": perplexity_results['num_samples'],
                "num_valid_samples": perplexity_results['num_valid_samples'],
                "std_perplexity": perplexity_results['statistics']['std_perplexity'],
                "min_perplexity": perplexity_results['statistics']['min_perplexity'],
                "max_perplexity": perplexity_results['statistics']['max_perplexity'],
               }'''
