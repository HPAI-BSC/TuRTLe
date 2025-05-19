"""
RTL-Repo: A Benchmark for Evaluating LLMs on Large-Scale RTL Design Projects
https://arxiv.org/pdf/2405.17378

The RTL-Repo benchmark provides a valuable resource for the hardware design
community to assess and compare LLMs’ performance in realworld RTL design scenarios
and train LLMs specifically for Verilog code generation in complex, multi-file RTL projects.

Homepage: https://github.com/AUCOHL/RTL-Repo
"""
import os
import re

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from metrics.code_eval import estimate_pass_at_k
from src.turtle_eval.base import TaskExtension
from transformers import AutoTokenizer

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
        #model, rtlrepo_use_train_ds, max_length_generation, path_data_benchmark, path_dataset_test, path_temporary_files):
        super().__init__(
            stop_words=[],
            requires_execution=False,
        )

        kwargs = kwargs.get("kwargs", {})

        self.path_dataset_test = kwargs.get("path_dataset_test")
        self.rtlrepo_use_train_ds = kwargs.get("rtlrepo_use_train_ds")
        self.model = kwargs.get("model")
        self.max_length_generation = kwargs.get("max_length_generation")
        
        print(f"Loading RTL-Repo dataset from {self.path_dataset_test}")
        print(f"Evaluating model {self.model}")
        print(f"Using train+test dataset? {self.rtlrepo_use_train_ds}")
        print(f"Max length generation: {self.max_length_generation}")

        self.max_length_prompt = self.max_length_generation - 80

        ds = load_from_disk(self.path_dataset_test)
        if self.rtlrepo_use_train_ds:
            self.dataset = concatenate_datasets([ds['train'], ds['test']])
        else:
            self.dataset = ds['test']

        # Use the sae tokenizer as the model being evaluated
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

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

        for snippet in doc['context']:
            cross_file_prompt += f"// Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"

        in_file_prompt = f"// Path: {doc['file_path']}\n{doc['cropped_code']}\n"

        if self.tokenizer is not None and self.max_length_prompt is not None:
            cross_file_prompt_token_nums = len(self.tokenizer.encode(cross_file_prompt))
            in_file_prompt_token_nums = len(self.tokenizer.encode(in_file_prompt))

            exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - self.max_length_prompt

            if exceed_token_nums > 0:
                cross_file_prompt_lines = cross_file_prompt.split("\n")
                for i in range(len(cross_file_prompt_lines)-1, -1, -1):
                    exceed_token_nums -= len(self.tokenizer.encode(cross_file_prompt_lines[i]))

                    if exceed_token_nums < 0:
                        break

                cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"

        prompt = cross_file_prompt + in_file_prompt
        prompt = re.sub(r'\n{4,}', '\n\n', prompt)

        # Truncate the prompt so that its tokenized version respects self.max_length_prompt
        # This procedure of the original benchmark fails as only truncates cross_file_prompt
        # which is not enough for samples with large in_file_prompt.
        # This is a problem, since truncating in_file_prompt results in a prompt that is not valid
        # to predict the next line. That's why we truncate the prompt from the beggining.
        tokenized = self.tokenizer(prompt, truncation=False)["input_ids"]
        truncated = tokenized[-self.max_length_prompt:] # truncate from the beggining
        prompt = self.tokenizer.decode(truncated, skip_special_tokens=True)

        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc['next_line']

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
        # generations could be a list of lists if n_samples > 1
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
        '''
        from fuzzywuzzy import fuzz
        edit_sim = 0.0
        if n_samples > 1:
            for idx, gt in enumerate(references):
                for sample in generations[idx]: # n_samples
                    edit_sim += fuzz.ratio(sample, gt)
        else: # n_samples == 1
            for pred, gt in zip(generations, references):
                edit_sim += fuzz.ratio(pred[0], gt)

        mean_es = round(edit_sim / total_samples, 5)'''

        #################
        # Calculate pass@k for exact_match
        #################
        if n_samples > 1:
            correct_em =[]
            for i in range(len(references)):
                n_correct_em = 0
                for j in range(len(generations[i])): # can be either 1 or 20 samples in our case
                    generation = generations[i][j]
                    result = 1 if generation.split() == references[i].split() else 0
                    n_correct_em += result
                correct_em.append(n_correct_em)

            ks = [1, 5]
            total = np.full(len(correct_em), n_samples)
            correct_em = np.array(correct_em)
            pass_em = {f"pass@{k}": estimate_pass_at_k(total, correct_em, k).mean()
                    for k in ks if (total >= k).all()}
        else: # n_samples == 1
            exact_match = 0
            for pred, gt in zip(generations, references):
                if pred[0].split() == gt.split():
                    exact_match += 1
            pass_em = round(exact_match / total_samples, 5)
        
        return {"exact_match": pass_em}
