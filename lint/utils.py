from .constants import MODEL_PATHS, INTERCEPTION_MAP, LOG_PATH

from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

import sys
import os
import json
import pickle
import string
import torch


def load_existing_results(result_path):
    results = pickle.load(open(f"{result_path}.pkl", "rb",))

    json_data = json.load(open(f"{result_path}.json",))

    # let's try to sync the two files
    for behavior in results:
        if behavior not in json_data:
            results[behavior] = []
            continue

        json_results = set(map(lambda r: r["content"], json_data[behavior]))
        results[behavior] = list(
            filter(lambda r: r["content"] in json_results, results[behavior])
        )

    results = dict(filter(lambda r: len(r[1]) > 0, results.items()))
    return results


def construct_prompt(behavior, magic_prompt):
    behavior = behavior.strip()
    prompt = f"Question: {behavior.rstrip(string.punctuation).strip()}.\nAnswer: "
    if magic_prompt != "none":
        prompt += magic_prompt.strip() + " "

    return prompt


def prepare_logger(data_dir, model, magic_prompt):
    log_path = os.path.join(data_dir, LOG_PATH)

    logger.remove()
    logger.add(
        sink=sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <green>%s[%s]</green> | <level>{level: <8}</level> | <level>{message}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        % (model, magic_prompt),
        level="DEBUG",
    )
    logger.add(
        log_path,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <green>%s[%s]</green> | <level>{level: <8}</level> | <level>{message}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        % (model, magic_prompt),
        level="DEBUG",
    )

    return logger


def is_contiguous_subsequence(subseq, seq):
    n = len(subseq)
    return any(subseq == seq[i : i + n] for i in range(len(seq) - n + 1))


def load_model(name, interception=True):
    if name in MODEL_PATHS:
        path = MODEL_PATHS[name]

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(path)

        setattr(model, "model_name", name)

        if interception:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(
                path, add_prefix_space=True
            )

            def get_tokens(word):
                return torch.tensor(
                    tokenizer_with_prefix_space(
                        [word], add_special_tokens=False
                    ).input_ids[0]
                ).to(model.device)

            interception_map = {}
            for _k, _v in INTERCEPTION_MAP.items():
                k, v = get_tokens(_k), get_tokens(_v)
                assert len(k) == len(
                    v
                ), f"token length mismatch: {_k} ({len(k)}) != {_v} ({len(v)})"
                interception_map[k] = v
            setattr(model, "interception_map", interception_map)

    else:
        assert False, "Unreachable code"

    return model, tokenizer
