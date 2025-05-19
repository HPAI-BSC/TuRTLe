import re
import warnings

from torch.utils.data import IterableDataset

INFILL_MODE = False
INSTRUCTION_MODE = False


class PrompBatcher(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        max_length,
        n_tasks,
        limit_start=0,
        prefix="",
        instruction_tokens=None,
        continuous_batching_size=None
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.prefix = prefix
        self.instruction_tokens = instruction_tokens
        if continuous_batching_size:
            if n_tasks > continuous_batching_size:
                self.continuous_batching_size = continuous_batching_size
            else: 
                self.continuous_batching_size = n_tasks
        else:
            self.continuous_batching_size = None

    def __iter__(self):
        prompts = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = self._make_infill_prompt(
                        **prompt_contents, preprefix=self.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE and INSTRUCTION_MODE:
            raise ValueError(
                "Cannot avail instruction following and infilling modes at once"
            )
        if self.continuous_batching_size:
            for start_index in range(0, len(prompts), self.continuous_batching_size):
                yield prompts[start_index: min(start_index+self.continuous_batching_size, len(prompts))]
        else: 
            yield prompts


    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def complete_code(
    task,
    model,
    tokenizer,
    prompt_batched_iterator,
    args,
    limit_start=0,
    prefix="",
    instruction_tokens=None,
    postprocess=True
):
    from vllm import SamplingParams

    # keep track of the list of generated codes
    # where len(code_gens) = n_tasks and len(code_gens[0]) = number of generated code samples
    code_gens = []
    batch_id = 0
    
    # Validate top_k value
    if hasattr(args, 'top_k') and args.top_k == 0:
        args.top_k = -1  # Set to -1 to disable top_k sampling
    
    for prompts in prompt_batched_iterator:
        print(f"Handling continuous batch {batch_id} of size {len(prompts)}.")
        if INFILL_MODE:
            # Treat eos token as a regular stop word not removing it from the output
            sampling_params = SamplingParams(
                n=args.n_samples,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=task.stop_words,
                skip_special_tokens=False,
                spaces_between_special_tokens=True
            )
        else:
            sampling_params = SamplingParams(
                n=args.n_samples,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens if args.max_tokens is not None else args.max_length_generation,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=task.stop_words
            )            
        batch_outputs = model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        for sample_output in batch_outputs:
            sample_gens = []
            for generation in sample_output.outputs:
                sample_gens.append(generation.text)
            code_gens.append(sample_gens)

        batch_id+=1

    return update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens
    )


def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens
):  
    updated_code_gens = []
    for sample_id, sample_gens in enumerate(code_gens):
        updated_sample_gens = []
        for generation in sample_gens:
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                if tokenizer.bos_token:
                    if generation.startswith(tokenizer.bos_token):
                        generation = generation[len(tokenizer.bos_token):]
                if tokenizer.eos_token:
                    if generation.startswith(tokenizer.eos_token):
                        generation = generation[len(tokenizer.eos_token):]
                if tokenizer.pad_token:
                    if generation.startswith(tokenizer.pad_token):
                        generation = generation[len(tokenizer.pad_token):]
                try:
                    # some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
                    tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
                    if generation.startswith(f"{tokenizer_prefix}"):
                        generation = generation[len(tokenizer_prefix):].lstrip()
                except:
                    pass
                if INFILL_MODE:
                    generation = _parse_infill(generation, tokenizer)
                if INSTRUCTION_MODE:
                    generation = _parse_instruction(generation, instruction_tokens)
            if not INFILL_MODE:
                generation = generation[len(prefix) :]
            if postprocess:
                updated_sample_gens.append(
                    task.postprocess_generation(generation, int(sample_id) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                updated_sample_gens.append(generation)
        updated_code_gens.append(updated_sample_gens)
    return updated_code_gens


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code
