from .interrogator import Interrogator
from .utils import load_existing_results, load_model, construct_prompt, prepare_logger

import argparse
import json
import os
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.utils.logging.get_logger("transformers").setLevel(
    transformers.utils.logging.ERROR
)


def parse_args():
    parser = argparse.ArgumentParser(description="Interogate LLMs for our own purpose.")

    parser.add_argument(
        "--model",
        type=str,
        default="llama2-13b",
        choices=[
            "llama2-7b",
            "llama2-13b",
            "llama2-70b",
            "yi",
            "vicuna",
            "mistral",
            "codellama-python",
            "codellama-instruct",
        ],
        help="the LLM we are going to attack",
    )

    parser.add_argument(
        "--eval-model",
        type=str,
        default="self",
        choices=[
            "none",
            "self",
            "llama2-7b",
            "llama2-13b",
            "llama2-70b",
            "vicuna",
            "mistral",
        ],
        help="the LLM we are going to use as an evaluator",
    )

    parser.add_argument(
        "--entailment-force-depth",
        type=int,
        default=1,
        help="the depth we are going to use entailment to force the model",
    )

    parser.add_argument(
        "--magic-prompt",
        type=str,
        default="none",
        help="the magic prompt we are going to use, `none` means no magic prompt (default is `none`)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="the batch size of each inference round",
    )

    parser.add_argument(
        "--searching-max-token-n",
        type=int,
        default=200,
        help="the max token number each interrogation result can have",
    )

    parser.add_argument(
        "--searching-topk",
        type=int,
        default=500,
        help="the top-k candidates we are considering",
    )

    parser.add_argument(
        "--searching-check-n",
        type=int,
        default=20,
        help="the number of sorted next-sentence candiates (by entailment score) we are going to check",
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="performing the interrogation process manually",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="overwriting existing results"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="stdin",
        help="the input file we are going to use, `stdin` means standard input (default is `stdin`)",
    )

    parser.add_argument(
        "--no-interception",
        action="store_true",
        help="no interception during the model generation",
    )

    parser.add_argument(
        "--target-n",
        type=int,
        default=5,
        help="the number of interrogated results we are going to search",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="the directory of the data we are going to use",
    )

    parser.add_argument(
        "--classifier-type",
        type=str,
        default="entailment",
        choices=["entailment", "gptfuzzer"],
        help="the ranking classifier we are going to use",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # log
    logger = prepare_logger(args.data_dir, args.model, args.magic_prompt)

    # read current results
    result_path = os.path.join(args.data_dir, f"{args.model}-results")
    if not args.overwrite and os.path.isfile(f"{result_path}.pkl"):
        results = load_existing_results(result_path)
    else:
        results = {}

    # get the subject model
    model, tokenizer = load_model(args.model, interception=not args.no_interception)

    # get the evaluator model
    if args.eval_model == "self":
        eval_model, eval_tokenizer = model, tokenizer
    elif args.eval_model == "none":
        eval_model, eval_tokenizer = None, None
    else:
        eval_model, eval_tokenizer = load_model(args.eval_model, interception=False)

    interrogator = Interrogator(
        model=model,
        tokenizer=tokenizer,
        logger=logger,
        eval_model=eval_model,
        eval_tokenizer=eval_tokenizer,
        searching_manual=args.manual,
        searching_topk=args.searching_topk,
        searching_check_n=args.searching_check_n,
        searching_entailment_force_depth=args.entailment_force_depth,
        searching_target_n=args.target_n,
        searching_max_token_n=args.searching_max_token_n,
        dump_tmp_results=True,
        batch_size=args.batch_size,
        interception=not args.no_interception,
        results=results,
        classifier_type=args.classifier_type,
    )

    if args.input == "stdin":
        while True:
            behavior = (
                input("Enter your instruction: ").encode().decode("unicode-escape")
            ).strip()

            if behavior == "/quit":
                exit(0)

            user_prompt = construct_prompt(behavior, args.magic_prompt)
            interrogator.interrogate(behavior, user_prompt)

    else:
        behaviors = list(json.load(os.path.join(args.data_dir, args.input)).keys())
        behaviors.sort()

        for behavior in behaviors:
            behavior = behavior.strip()
            user_prompt = construct_prompt(behavior, args.magic_prompt)
            interrogator.interrogate(behavior, user_prompt)

        interrogator.dump_results(result_path)


if __name__ == "__main__":
    main()
