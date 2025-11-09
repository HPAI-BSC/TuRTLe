import itertools

import numpy as np
import math

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def compute_perplexities(llm, sampling_params, references, dataset, get_prompt, max_model_len):

    vllm_tok = llm.get_tokenizer()
    perplexities = []

    for i, doc in enumerate(dataset):
        prompt = get_prompt(doc)
        ref = references[i]

        full_text = prompt + ref

        # Tokenize prompt and full text to identify response token positions
        response_tokens = vllm_tok.encode(ref, add_special_tokens=False)
        full_tokens = vllm_tok.encode(full_text, add_special_tokens=False)

        # ensure max_model_len is respected
        if len(full_tokens) > max_model_len:
            print(f"WARNING: prompt+response exceeds max_model_len. Truncating from the left...")
            # Truncate from the left to maintain the response at the end
            full_tokens = full_tokens[-max_model_len:]

        # print(f"len(response_tokens): {len(response_tokens)}")
        # print(f"len(full_tokens): {len(full_tokens)}")

        # Find where response tokens start in the full text
        response_start_idx = len(full_tokens) - len(response_tokens)
        
        outputs = llm.generate({"prompt_token_ids":full_tokens}, sampling_params)
        output = outputs[0] # There is only one output
        # print(output.prompt_logprobs) # list of #full_tokens elements, each element is a dict with key "token_id" and value "Logprob"
        # Value example: Logprob(logprob=-3.3348352909088135, rank=3, decoded_token='110')
        # print(f"response_tokens: {response_tokens}")
        # print(f"output.prompt_logprobs[response_start_idx:]: {output.prompt_logprobs[response_start_idx:]}")

        # Extract log probabilities only for response tokens
        response_log_probs = []
        response_logprobs = output.prompt_logprobs[response_start_idx:] # each element is a dict with key "token_id" and value "Logprob"
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
            
    valid_perplexities = [p for p in perplexities if not math.isinf(p) and p<1000]

    return {
        "num_samples": len(references),
        "num_valid_samples": len(valid_perplexities),
        "statistics": {
                        "mean_perplexity": np.mean(valid_perplexities),
                        "median_perplexity": np.median(valid_perplexities),
                        "std_perplexity": np.std(valid_perplexities),
                        "min_perplexity": np.min(valid_perplexities),
                        "max_perplexity": np.max(valid_perplexities),
                        },
        "perplexities": perplexities
    }

def compute_min_k_metrics(logprobs, ks):
    """Helper function to compute min-k metrics for a list of logprobs"""
    sorted_logprobs = sorted(logprobs)
    n_tokens = len(logprobs)
    
    def index_cut(k, n_tokens):
        return int(np.ceil(n_tokens * (k / 100.0)))
    
    results = {}
    for k in ks:
        cut = index_cut(k, n_tokens)
        mean = np.mean(sorted_logprobs[:cut])
        results[f'k={k}'] = mean
    
    return results

def compute_min_k(llm, sampling_params, references, dataset, get_prompt, max_model_len, ks):

    vllm_tok = llm.get_tokenizer()
    response_min_ks_no_context = []
    prompt_min_ks = []
    response_min_ks = []
    full_min_ks = []

    for i, doc in enumerate(dataset):
        prompt = get_prompt(doc)
        ref = references[i]

        full_text = prompt + ref

        # Tokenize prompt and full text to identify response token positions
        response_tokens = vllm_tok.encode(ref, add_special_tokens=False)
        full_tokens = vllm_tok.encode(full_text, add_special_tokens=False)

        # ensure max_model_len is respected
        if len(full_tokens) > max_model_len:
            print(f"WARNING: prompt+response exceeds max_model_len. Truncating from the left...")
            # Truncate from the left to maintain the response at the end
            full_tokens = full_tokens[-max_model_len:]

        #################
        # Response (no context)
        #################

        outputs = llm.generate({"prompt_token_ids":response_tokens}, sampling_params)
        output = outputs[0] # There is only one output

        # Extract logprobs
        logprobs = []
        for i, token_logprobs in enumerate(output.prompt_logprobs):
            if i > 0: # logprob of first prompt token is None
                logprob_value = token_logprobs[response_tokens[i]].logprob
                logprobs.append(logprob_value)

        response_min_ks_no_context.append(compute_min_k_metrics(logprobs, ks))

        #################
        # Prompt + Response (with context) + full text
        #################
            
        response_start_idx = len(full_tokens) - len(response_tokens)

        outputs = llm.generate({"prompt_token_ids":full_tokens}, sampling_params)
        output = outputs[0] # There is only one output

        # Extract logprobs for all tokens
        all_logprobs = []
        for i, token_logprobs in enumerate(output.prompt_logprobs):
            if i > 0: # logprob of first prompt token is None
                logprob_value = token_logprobs[full_tokens[i]].logprob
                all_logprobs.append(logprob_value)

        # Split logprobs into prompt and response portions
        prompt_logprobs = all_logprobs[:response_start_idx-1] # -1 because first logprob is None
        response_logprobs = all_logprobs[response_start_idx-1:]

        # Calculate Min-K% metrics for each portion
        prompt_min_ks.append(compute_min_k_metrics(prompt_logprobs, ks))
        response_min_ks.append(compute_min_k_metrics(response_logprobs, ks))
        full_min_ks.append(compute_min_k_metrics(all_logprobs, ks))

        # Proactively release any cached VRAM between requests
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Helper function to compute mean statistics
    def compute_stats(min_ks_list, ks):
        stats = {}
        for k in ks:
            values = [s[f'k={k}'] for s in min_ks_list]
            stats[f'mean k={k}'] = np.mean(values)
            stats[f'median k={k}'] = np.median(values)
        return stats

    return {
        "prompt": compute_stats(prompt_min_ks, ks),
        "response (with context)": compute_stats(response_min_ks, ks),
        "response (no context)": compute_stats(response_min_ks_no_context, ks),
        "full_text":compute_stats(full_min_ks, ks)
    }

def compute_min_k_veriContaminated(llm, sampling_params, references, dataset, get_prompt, max_model_len=None):

    ks = [10, 15, 20, 25, 30, 40, 60, 80, 100]
    
    vllm_tok = llm.get_tokenizer()
    response_min_ks_no_context = []
    prompt_min_ks = []
    response_min_ks = []
    full_min_ks = []

    for i, doc in enumerate(dataset):
        prompt = get_prompt(doc)
        ref = references[i]

        full_text = prompt + ref
        print(f'-----------------------------BEGINNING SAMPLE {i}-----------------------------')
        print(prompt)
        print('####################################')
        print(ref)
        print(f'-----------------------------END SAMPLE {i}-----------------------------')

        # Tokenize prompt and full text to identify response token positions
        response_tokens = vllm_tok.encode(ref, add_special_tokens=False)
        full_tokens = vllm_tok.encode(full_text, add_special_tokens=False)

        # ensure max_model_len is respected
        if max_model_len and len(full_tokens) > max_model_len:
            print(f"WARNING: prompt+response exceeds max_model_len. Truncating from the left...")
            # Truncate from the left to maintain the response at the end
            full_tokens = full_tokens[-max_model_len:]

        #################
        # Response (no context)
        #################

        outputs = llm.generate({"prompt_token_ids":response_tokens}, sampling_params)
        output = outputs[0] # There is only one output

        # Extract logprobs
        logprobs = []
        for i, token_logprobs in enumerate(output.prompt_logprobs):
            if i > 0: # logprob of first prompt token is None
                logprob_value = token_logprobs[response_tokens[i]].logprob
                logprobs.append(math.exp(logprob_value))

        response_min_ks_no_context.append(compute_min_k_metrics(logprobs, ks))

        #################
        # Prompt + Response (with context) + full text
        #################
            
        response_start_idx = len(full_tokens) - len(response_tokens)

        outputs = llm.generate({"prompt_token_ids":full_tokens}, sampling_params)
        output = outputs[0] # There is only one output

        # Extract logprobs for all tokens
        all_probs = []
        for i, token_logprobs in enumerate(output.prompt_logprobs):
            if i > 0: # logprob of first prompt token is None
                logprob_value = token_logprobs[full_tokens[i]].logprob
                all_probs.append(math.exp(logprob_value)) # get probabilities

        # Split logprobs into prompt and response portions
        prompt_probs = all_probs[:response_start_idx-1] # -1 because first logprob is None
        response_probs = all_probs[response_start_idx-1:]

        # Calculate Min-K% metrics for each portion
        prompt_min_ks.append(compute_min_k_metrics(prompt_probs, ks))
        response_min_ks.append(compute_min_k_metrics(response_probs, ks))
        full_min_ks.append(compute_min_k_metrics(all_probs, ks))

        # Proactively release any cached VRAM between requests
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute contamination rates for different thresholds
    def get_contamination_rates(min_ks):
        n = len(min_ks) # number of samples
        thresholds = np.linspace(0, 1, num=10)
        rates = []
        for th in thresholds:
            contaminated_samples = len([s for s in min_ks if s>th])
            rates.append(contaminated_samples/n)
        return rates

    # Helper function to compute statistics
    def compute_stats(min_ks_list, ks):
        stats = {}
        for k in ks:
            values = [s[f'k={k}'] for s in min_ks_list]
            stats[f'mean k={k}'] = np.mean(values)
            stats[f'median k={k}'] = np.median(values)
            stats[f'contamination_rates k={k}'] = get_contamination_rates(values)
        return stats

    return {
        "prompt": compute_stats(prompt_min_ks, ks),
        "response (with context)": compute_stats(response_min_ks, ks),
        "response (no context)": compute_stats(response_min_ks_no_context, ks),
        "full text":compute_stats(full_min_ks, ks)
    }