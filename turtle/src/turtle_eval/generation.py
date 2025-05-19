import json
import torch
import numpy as np
import random


from .utils import (
    PrompBatcher, 
    complete_code
)

def set_seed(seed=77)-> None:
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_generations(
        task,
        dataset,
        model,
        tokenizer,
        n_tasks,
        args
    ):
    """
    Generate code using the provided model and tokenizer.
    Parameters:
        - task (str): The task name.
        - dataset (Dataset): The dataset to use for generation.
        - model (Model): The model to use for generation.
        - tokenizer (Tokenizer): The tokenizer to use for generation.
        - n_tasks (int): The number of tasks to generate code for.
        - args (Namespace): The command-line arguments.
    Returns:
        - list: A list of generated code.
    """
    set_seed(seed=args.seed)
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            print(
                f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
            )
        return generations[:n_tasks]

    if task.stop_words and tokenizer.eos_token:
        task.stop_words.append(tokenizer.eos_token)    

    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None

    print(f"number of problems for this task is {n_tasks}")

    prompt_batched_iterator = PrompBatcher(
        task,
        dataset,
        tokenizer,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        limit_start=args.limit_start,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        continuous_batching_size=args.continuous_batching_size
    )

    generations = complete_code(
        task,
        model,
        tokenizer,
        prompt_batched_iterator,
        args=args,
        limit_start=args.limit_start,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess
    )
    return generations
