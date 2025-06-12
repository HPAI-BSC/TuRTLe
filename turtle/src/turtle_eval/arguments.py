import fnmatch
from dataclasses import dataclass, field
from typing import Optional

try:
    from bigcode_eval.arguments import EvalArguments
except ModuleNotFoundError: # the evaluation image does not have the pypi package
    @dataclass
    class EvalArguments:
        """
        Configuration for running the evaluation.
        """
        prefix: Optional[str] = field(
            default="",
            metadata={
                "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
            },
        )
        do_sample: Optional[bool] = field(
            default=True,
            metadata={"help": "Sample from the language model's output distribution."},
        )
        temperature: Optional[float] = field(
            default=0.2, metadata={"help": "Sampling temperature used for generation."}
        )
        top_k: Optional[int] = field(
            default=0, metadata={"help": "Top-k parameter used for generation."}
        )
        top_p: Optional[float] = field(
            default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
        )
        n_samples: Optional[int] = field(
            default=1,
            metadata={"help": "Number of completions to generate for each sample."},
        )
        eos: Optional[str] = field(
            default="<|endoftext|>", metadata={"help": "end of sentence token."}
        )
        seed: Optional[int] = field(
            default=0, metadata={"help": "Random seed used for evaluation."}
        )

from utils.task_updater import TaskUpdater


# Add lazy initialization of task_updater
def get_task_updater() -> TaskUpdater:
    """Lazy initialization of TaskUpdater to avoid circular imports"""
    # rprint folder on sys.path

    task_updater = TaskUpdater()
    task_updater.apply_updates()
    return task_updater


def pattern_match(patterns: list, source_list: list) -> str:
    """Returns a list containing all values of the source_list that
    match at least one of the patterns
    Parameters:
        - patterns (list): List of patterns to match
        - source_list (list): List of strings to match against
    Returns:
        - str: The first matching string from the source_list
    """
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    if len(task_names) > 1:
        raise ValueError(
            f"This repo only supports one task at a time but received {len(task_names)} tasks"
        )
    return list(task_names)[0]


@dataclass
class ExtendedModelArgument(EvalArguments):
    """
    Extended configuration for evaluation with additional parameters.
    Inherits from the original EvalArguments and adds new functionality.
    """

    # Model Arguments
    model: str = field(
        default=None,
        metadata={
            "help": "Model to evaluate, provide a repo name in Hugging Face hub or a local path"
        },
    )
    use_auth_token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use the token generated when running `huggingface-cli login` (necessary for private model)."
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use a model with custom code, this requires executing code by the author of the model."
        },
    )
    precision: Optional[str] = field(
        default="fp32", metadata={"help": "Model precision, from: fp32, fp16 or bf16"}
    )
    left_padding: Optional[bool] = field(
        default=False,
        metadata={"help": "Force left padding, needed for models like chatglm3-6b"},
    )


@dataclass
class ExtendedWorkflowArguments(EvalArguments):
    """
    Extended configuration for workflow with additional parameters.
    Inherits from the original EvalArguments and adds new functionality.
    """

    tasks: str = field(
        default=None,
        metadata={
            "help": lambda: f"Evaluation tasks from {get_task_updater().ALL_TASKS}"
        },
    )
    instruction_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": "A series of instruction tokens used for instruction-tuning benchmarks"
            + " separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>"
        },
    )
    metric_output_path: Optional[str] = field(
        default="/tmp/evaluation_results.json",
        metadata={"help": "Path to save the results"},
    )
    save_generations: Optional[bool] = field(
        default=True, metadata={"help": "Whether to save code generations"}
    )
    save_generations_path: Optional[str] = field(
        default="/tmp/generations.json",
        metadata={"help": "Path for saving the code generations"},
    )
    save_references: Optional[bool] = field(
        default=True, metadata={"help": "Whether to save reference solutions/tests"}
    )
    save_references_path: Optional[str] = field(
        default="/tmp/references.json",
        metadata={"help": "Path for saving the references solutions/tests"},
    )
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt type to use for generation in HumanEvalPack tasks"},
    )
    limit: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to solve and evaluate from the benchmark"},
    )
    limit_start: Optional[int] = field(
        default=0,
        metadata={
            "help": "Optional offset to start from when limiting the number of samples"
        },
    )
    postprocess: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Postprocess model outputs before execution, always on except during generation tests"
        },
    )
    allow_code_execution: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Allow code evaluation to execute external/untrusted Python code on your machine"
        },
    )
    generation_only: Optional[bool] = field(
        default=False, metadata={"help": "Do code generation but no evaluation"}
    )
    load_generations_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path of file with previously generated solutions, if provided generation is"
            + "skipped and only evaluation is done"
        },
    )
    load_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path of additional data to load for the tasks"}
    )
    rtlrepo_use_train_ds: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use train+test dataset for the RTL-Repo benchmark."
            + "If False, only the test dataset is used."
        },
    )
    few_shot: Optional[int] = field(
        default=0, metadata={"help": "Fewshot param for VerilogEval tasks ICL."}
    )
    syntax_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path of file with previously generated solutions, if provided generation is"
            + "skipped and only evaluation is done"
        },
    )
    path_data_benchmark: Optional[str] = field(
        default=None, metadata={"help": "Path to the benchmark data directory"}
    )
    path_dataset_test: Optional[str] = field(
        default=None, metadata={"help": "Path to the test dataset"}
    )
    path_temporary_files: Optional[str] = field(
        default=None, metadata={"help": "Path for temporary files storage"}
    )
    base_path_models: Optional[str] = field(
        default=None, metadata={"help": "Path to the models directory"}
    )
    max_length_generation: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of generated sequence (prompt+generation)"},
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens to generate per output sequence"},
    )
    rtllm_version: Optional[str] = field(
        default="2.0",
        metadata={"help": "RTLLM benchmark version. Can be either 2.0 or 1.1"},
    )

    def __post_init__(self):
        # Access to metadata to force theevaluation of the lambda function
        if callable(self.__dataclass_fields__["tasks"].metadata["help"]):
            self.__dataclass_fields__["tasks"].metadata["help"] = (
                self.__dataclass_fields__["tasks"].metadata["help"]()
            )
        # if hasattr(self, "file_handler") and self.file_handler is not None:
        #     task_updater = get_task_updater(self.file_handler)
        #     self.__dataclass_fields__["tasks"].metadata["help"] = (
        #         f"Evaluation tasks from {task_updater.ALL_TASKS}"
        #     )


@dataclass
class ExtendedVLLMArguments(EvalArguments):
    """
    Extended configuration for VLLM with additional parameters.
    Inherits from the original EvalArguments and adds new functionality.
    """

    # VLLM Arguments
    gpu_memory_utilization: Optional[float] = field(
        default=0.95, metadata={"help": "Proportion of GPU memory to reserve for vllm"}
    )
    swap_space: Optional[int] = field(
        default=64, metadata={"help": "RAM memory to reserve for excess GPU pages"}
    )
    continuous_batching_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of dataset samples to be sent at a time for vllm to apply continuous batching. "
            + "If None (default), all the prompts are sent to the LLM Engine together. Make sure to "
            + "modify the CPU swap_space as you modify this parameter or you may get OOM errors."
        },
    )
    sequence_length_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The longest length of the models forward graph to capture when running the VLLM.LLM Engine. "
            + "Behind the scenes, this argument controls both, the max_seq_len_to_capture and the max_model_len "
            + "parameters to the LLM class. Can be used to control the sequence length of a model whose default supported "
            + "sequence length is very large such as Llama-3.1 or Qwen-2"
        },
    )
    tensor_parallel_size: Optional[int] = field(
        default=4, metadata={"help": "Number of tensor parallel replicas."}
    )
    ip: Optional[str] = field(
        default=None,
        metadata={"help":"IP of the vLLM inference server."}
    )
    port: Optional[str] = field(
        default=None,
        metadata={"help":"Port of the vLLM inference server."}
    )



@dataclass
class ExtendedGenerationArguments(EvalArguments):
    """
    Extended configuration for generation with additional parameters.
    Inherits from the original EvalArguments and adds new functionality.
    """

    # Generation Arguments (overriding some parent defaults)
    temperature: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Sampling temperature used for generation. "
            + "Temperatures lower than 1e-5 will leads to switching to greedy mode."
        },
    )
    top_k: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Top-k parameter used for generation. Disabled (-1) by default. "
            + "Set to an integer of at least 1 to enable."
        },
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether "
            + "they appear in the prompt and the generated text so far. Values > 1 "
            + "encourage the model to use new tokens, while values < 1 encourage "
            + "the model to repeat tokens."
        },
    )
    frequency_penalty: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on their "
            + "frequency in the generated text so far. Values > 0 encourage the "
            + "model to use new tokens, while values < 0 encourage the model to "
            + "repeat tokens."
        },
    )
    presence_penalty: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they "
            + "appear in the generated text so far. Values > 0 encourage the model "
            + "to use new tokens, while values < 0 encourage the model to repeat"
            + "tokens."
        },
    )

