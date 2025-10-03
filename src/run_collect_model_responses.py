import argparse
import json
import os
import time

from model_response_collection.collect_model_responses import ResponseCollector
from prompting.prompting_strategy import PromptingStrategy
from utils import get_dataset, get_language_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')
    parser.add_argument('--example_idxs', type=int, nargs='+', default=[], help='indices of examples to analyze. If empty, all examples will be analyzed')
    parser.add_argument('--example_idx_start', type=int, default=0, help='index of first example to analyze (if example_idxs not specified, and all examples are to be analyzed)')
    parser.add_argument('--n_examples', type=int, help='number of examples to analyze (if example_idxs not specified)')
    parser.add_argument('--language_model', type=str, default='gpt-3.5-turbo-instruct', help='name of language model to collect responses from')
    parser.add_argument('--language_model_max_tokens', type=int, default=256, help='max tokens for language model. Only relevant for completion GPT (since default max tokens is inf for Chat GPT.')
    parser.add_argument('--language_model_temperature', type=float, default=0.7, help='temperature to use for language model when collecting responses')
    parser.add_argument('--cot', action='store_true', help='whether to use CoT when prompting model to analyze')
    parser.add_argument('--few_shot', action='store_true', help='whether to use few shot when prompting model to analyze')
    parser.add_argument('--knn_rank', action='store_true', help='whether to use knn rank when prompting model to analyze (for now, only applicable to MedQA)')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_prompt_qa', help='name of few shot prompt to use')
    parser.add_argument('--add_instr', type=str, default=None, help='additional instructions to add to prompt')
    parser.add_argument('--original_only', action='store_true', help='whether to only collect responses to the original examples')
    parser.add_argument('--n_completions', type=int, default=5, help='number of completions to generate for each intervention')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_workers', type=int, help='number of workers to use for parallel processing')
    parser.add_argument('--verbose', action='store_true', help='whether to print progress')
    parser.add_argument('--debug', action='store_true', help='whether to run in debug mode')
    parser.add_argument('--intervention_data_path', type=str, default='outputs/bbq/intervention-generation', help='path to directory with intervention data to use as inputs to model')
    parser.add_argument('--output_dir', type=str, default='outputs/bbq/faithfulness_measurement', help='path to directory to save results')
    parser.add_argument('--save_failed_responses', action='store_true', help='whether to save failed responses')
    parser.add_argument('--fresh_start', action='store_true', help='whether to start from scratch (i.e. not restart from previous run)')
    return parser.parse_args()


def validate_args(args):
    """
    Validates the arguments.
    Args: command line arguments parsed with the argparse library
    """
    if args.dataset != 'medqa':
        assert not args.knn_rank, "KNN Rank prompting strategy has only been implemented for MedQA. Please set knn_rank to False."


def collect_model_responses(dataset, cnt, example_idx, language_model, prompting_strategy, args, failure_dict):
    init_time = time.time()
    # sub directory within output directory for this example
    example_dir = os.path.join(args.output_dir, f"example_{example_idx}")
    # init model response collector
    rc = ResponseCollector(
        dataset=dataset,
        example_idx=example_idx,
        intervention_data_path=args.intervention_data_path,
        language_model=language_model,
        prompt_strategy=prompting_strategy,
        output_dir=example_dir,
        n_completions=args.n_completions,
        seed=args.seed,
        n_workers=args.n_workers,
        verbose=args.verbose,
        debug=args.debug,
        save_failed_responses=args.save_failed_responses,
        restart_from_previous=not args.fresh_start
    )
    # collect model responses to the original examples
    if args.dataset == "bbq" or args.dataset == "motivating-example":
        rc.collect_original_model_responses()
    else:
        rc.collect_original_model_responses()
    if args.original_only:
        failure_dict[example_idx] = rc.failures
        print(f"FINISHED COLLECTING ORIGINAL MODEL RESPONSES for example {example_idx} ({cnt} out of {len(args.example_idxs)}) in {time.time() - init_time} seconds\n\n")
        return
    # collect model responses to each intervened example
    rc.collect_counterfactual_model_responses()
    failure_dict[example_idx] = rc.failures
    print(f"FINISHED COLLECTING MODEL RESPONSES for example {example_idx} ({cnt} out of {len(args.example_idxs)}) in {time.time() - init_time} seconds\n\n")



def main():
    args = parse_args()
    print("ARGS...")
    print(args)
    validate_args(args)
    # init dataset
    dataset = get_dataset(args.dataset, args.dataset_path)
    # init language model
    language_model = get_language_model(args.language_model, max_tokens=args.language_model_max_tokens, temperature=args.language_model_temperature)
    # init prompting strategy
    prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if len(args.example_idxs) == 0:
        end_idx = len(dataset) if args.n_examples is None else args.example_idx_start + args.n_examples
        args.example_idxs = range(args.example_idx_start, end_idx)
    failed_idxs = dict()
    for cnt, example_idx in enumerate(args.example_idxs):
        try:
            collect_model_responses(dataset, cnt + 1, example_idx, language_model, prompting_strategy, args, failed_idxs)
        except Exception as e:
            print(f"ERROR: {e}")
            failed_idxs[example_idx] = str(e)
    # save failed examples
    with open(os.path.join(args.output_dir, 'failed_examples.json'), 'w') as f:
        json.dump(failed_idxs, f)


if __name__ == '__main__':
    main()