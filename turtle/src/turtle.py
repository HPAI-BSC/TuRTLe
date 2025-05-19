import json
import os
from dataclasses import fields, make_dataclass
from typing import Any, List, Type

import datasets
import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser
from turtle_eval.arguments import (ExtendedGenerationArguments,
                                   ExtendedModelArgument,
                                   ExtendedVLLMArguments,
                                   ExtendedWorkflowArguments, get_task_updater,
                                   pattern_match)
from turtle_eval.proxy import LCaseEvaluatorProxy


def parse_extended_args(extended_classes: List[Type[Any]]):
    """
    Robust argument parser that handles inherited arguments without conflicts.

    Args:
        extended_classes: List of dataclass types that inherit from EvalArguments

    Returns:
        Namespace with all parsed arguments
    """
    try:
        from bigcode_eval.arguments import EvalArguments
    except ModuleNotFoundError:
        from turtle_eval.arguments import EvalArguments

    # 1. Collect all unique fields from all classes
    fields_dict = {}
    for cls in [EvalArguments] + extended_classes:
        for field in fields(cls):
            if (
                field.name not in fields_dict
            ):  # Child classes override parent fields
                fields_dict[field.name] = field

    # 2. Dynamically create a combined dataclass
    CombinedConfig = make_dataclass(
        "CombinedConfig",
        [(f.name, f.type, f) for f in fields_dict.values()],
        bases=(EvalArguments,),
    )
    # 3. Parse all arguments at once
    parser = HfArgumentParser(CombinedConfig)

    args = parser.parse_args()

    return args


def main():
    # Initialize the TaskUpdater
    task_updater = get_task_updater()

    args = parse_extended_args(
        [
            ExtendedGenerationArguments,
            ExtendedModelArgument,
            ExtendedVLLMArguments,
            ExtendedWorkflowArguments,
        ]
    )

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    task = pattern_match(args.tasks.split(","), task_updater.ALL_TASKS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        evaluator = LCaseEvaluatorProxy(
            use_modified=True, model=None, tokenizer=None, args=args
        )
        print("evaluation only mode")
        results[task] = evaluator.evaluate(task)
    else:
        from vllm import LLM

        # here we generate code and save it (evaluation is optional but True by default)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )

        if args.left_padding:
            # left padding is required for some models like chatglm3-6b
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                padding_side="left",
            )
        else:
            # used by default for most models
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                truncation_side="left",
                padding_side="right",
            )
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        try:
            tokenizer.pad_token = tokenizer.eos_token

        # Some models like CodeGeeX2 have pad_token as a read-only property
        except AttributeError:
            print("Not setting pad_token to eos_token")
            pass
        WIZARD_LLAMA_MODELS = [
            "WizardLM/WizardCoder-Python-34B-V1.0",
            "WizardLM/WizardCoder-34B-V1.0",
            "WizardLM/WizardCoder-Python-13B-V1.0",
        ]
        if args.model in WIZARD_LLAMA_MODELS:
            tokenizer.bos_token = "<s>"
            tokenizer.bos_token_id = 1
            print("Changing bos_token to <s>")

        model = LLM(
            model=args.model,
            tensor_parallel_size=4,
            trust_remote_code=args.trust_remote_code,
            dtype=dict_precisions[args.precision],
            gpu_memory_utilization=args.gpu_memory_utilization,
            swap_space=args.swap_space,
            max_seq_len_to_capture=args.sequence_length_limit,
            max_model_len=args.sequence_length_limit,
        )
        model.set_tokenizer(tokenizer=tokenizer)

        evaluator = LCaseEvaluatorProxy(
            use_modified=True, model=model, tokenizer=tokenizer, args=args
        )

        if args.generation_only:
            print("generation mode only")
            generations, references = evaluator.generate_text(task)
            evaluator._save_json_files(
                generations,
                references,
                args.save_generations_path,
                args.save_references_path,
            )
        else:
            results[task] = evaluator.evaluate(task)

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        print(dumped)
        os.makedirs(os.path.dirname(args.metric_output_path), mode=755, exist_ok=True)
        with open(args.metric_output_path, "w+") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
