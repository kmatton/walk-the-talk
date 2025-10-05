import argparse
import json
import os
import time

from implied_concept_determination.determine_implied_concepts import ExplanationAnalyzer
from utils import get_dataset, get_language_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')
    parser.add_argument('--example_idxs', type=int, nargs='+', default=[], help='indices of examples to analyze. If empty, all examples will be analyzed')
    parser.add_argument('--example_idx_start', type=int, default=0, help='index of first example to analyze (if example_idxs not specified, and all examples are to be analyzed)')
    parser.add_argument('--n_examples', type=int, help='number of examples to analyze (if example_idxs not specified)')
    parser.add_argument('--implied_concepts_model', type=str, default='gpt-4o', help='name of language model to use determine implied concepts')
    parser.add_argument('--implied_concepts_model_max_tokens', type=int, default=256, help='max tokens for language model. Only relevant for completion GPT (since default max tokens is inf for Chat GPT.')
    parser.add_argument('--implied_concepts_model_temperature', type=float, default=0, help='temperature for language model used to determine implied concepts')
    parser.add_argument('--implied_concepts_model_n_completions', type=int, default=1, help='number of completions to generate for language model used to determine implied concepts')
    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt', help='name of base prompt for implied concepts determination step')
    parser.add_argument('--original_only', action='store_true', help='whether to run only on original questions')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers to use for parallel processing')
    parser.add_argument('--verbose', action='store_true', help='whether to print progress')
    parser.add_argument('--debug', action='store_true', help='whether to run in debug mode')
    parser.add_argument('--intervention_data_path', type=str, default='outputs/bbq/intervention-generation', help='path to directory with intervention data')
    parser.add_argument('--model_response_data_path', type=str, default='outputs/bbq/model-response', help='path to directory with model responses to intervened examples')
    parser.add_argument('--output_dir', type=str, default='outputs/bbq/implied_concepts', help='path to directory to save results')
    parser.add_argument('--fresh_start', action='store_true', help='whether to start from scratch (i.e. not restart from previous run)')
    return parser.parse_args()


def determine_implied_concepts(dataset, cnt, example_idx, implied_concepts_model, args, failure_dict):
    init_time = time.time()
    # sub directory within output directory for this example
    example_dir = os.path.join(args.output_dir, f"example_{example_idx}")
    # init explanation analyzer
    ea = ExplanationAnalyzer(
        dataset=dataset,
        example_idx=example_idx,
        implied_concepts_model=implied_concepts_model,
        implied_concepts_base_prompt_name=args.implied_concepts_base_prompt_name,
        intervention_data_path=args.intervention_data_path,
        model_response_path=args.model_response_data_path,
        output_dir=example_dir,
        n_completions=args.implied_concepts_model_n_completions,
        seed=args.seed,
        n_workers=args.n_workers,
        verbose=args.verbose,
        debug=args.debug,
        restart_from_previous=not args.fresh_start
    )
    # analyze model responses to the original examples
    ea.identify_concepts_implied_by_explanation("original")
    if args.original_only:
        failure_dict[example_idx] = ea.failures
        print(f"FINISHED DETERMINING IMPLIED FACTORS FOR ORIGINAL MODEL RESPONSES for example {example_idx} ({cnt} out of {len(args.example_idxs)}) in {time.time() - init_time} seconds\n\n")
        return
    # identify which concepts were implied by the explanation for counterfactual questions
    ea.identify_concepts_implied_by_explanation("counterfactual")
    failure_dict[example_idx] = ea.failures
    print(f"FINISHED DETERMINING IMPLIED FACTORS for example {example_idx} ({cnt} out of {len(args.example_idxs)}) in {time.time() - init_time} seconds\n\n")


def main():
    args = parse_args()
    print("ARGS...")
    print(args)
    # init dataset
    dataset = get_dataset(args.dataset, args.dataset_path)
    # init language model used for implied concepts
    implied_concepts_model = get_language_model(args.implied_concepts_model, max_tokens=args.implied_concepts_model_max_tokens, temperature=args.implied_concepts_model_temperature)
    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if len(args.example_idxs) == 0:
        end_idx = len(dataset) if args.n_examples is None else args.example_idx_start + args.n_examples
        args.example_idxs = range(args.example_idx_start, end_idx)
    failed_idxs = dict()
    for cnt, example_idx in enumerate(args.example_idxs):
        try:
            determine_implied_concepts(dataset, cnt + 1, example_idx, implied_concepts_model, args, failed_idxs)
        except Exception as e:
            print(f"ERROR: {e}")
            failed_idxs[example_idx] = str(e)
    # save failed examples
    with open(os.path.join(args.output_dir, 'failed_examples.json'), 'w') as f:
        json.dump(failed_idxs, f)
 

if __name__ == '__main__':
    main()