import argparse
import json
import time
import os

from intervention_generation.generate_interventions import InterventionGenerator
from utils import get_dataset, get_language_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')
    parser.add_argument('--example_idxs', type=int, nargs='+', default=[], help='indices of examples to analyze. If empty, all examples will be analyzed')
    parser.add_argument('--example_idx_start', type=int, default=0, help='index of first example to analyze (if example_idxs not specified, and all examples are to be analyzed)')
    parser.add_argument('--n_examples', type=int, help='number of examples to analyze (if example_idxs not specified)')
    parser.add_argument('--intervention_model', type=str, default='gpt-4o', help='name of model to use to generate counterfactuals data')
    parser.add_argument('--intervention_model_max_tokens', type=int, default=256, help='max tokens for LLM-based counterfactual generation model. Only relevant for completion GPT (since default max tokens is inf for Chat GPT).')
    parser.add_argument('--intervention_model_temperature', type=float, default=0, help='temperature for language model used for counterfactual example generation steps')
    parser.add_argument('--concept_id_only', action='store_true', help='whether to only run concept ID step (no concept values ID or counterfactual generation)')
    parser.add_argument('--concept_id_base_prompt_name', type=str, default='concept_id_prompt', help='name of base prompt for concept ID step')
    parser.add_argument('--concept_values_only', action='store_true', help='whether to only run only up to the concept values ID step (no counterfactual generation)')
    parser.add_argument('--concept_values_base_prompt_name', type=str, default='concept_values_prompt', help='name of base prompt for concept values ID step')
    parser.add_argument('--counterfactual_gen_base_prompt_name', type=str, default='counterfactual_gen_prompt', help='name of base prompt for counterfactual generation step')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers to use for parallel processing')
    parser.add_argument('--verbose', action='store_true', help='whether to print progress')
    parser.add_argument('--debug', action='store_true', help='whether to run in debug mode')
    parser.add_argument('--include_unknown_concept_values', action='store_true', help='whether to include unknown as a concept value in the intervention generation')
    parser.add_argument('--only_concept_removals', action='store_true', help='whether to only generate interventions that remove concepts')
    parser.add_argument('--fresh_start', action='store_true', help='whether to start from scratch (i.e. not restart from previous run)')
    return parser.parse_args()


def validate_args(args):
    """
    Validates the arguments.
    Args: command line arguments parsed with the argparse library
    """
    assert not (args.concept_values_only and args.concept_id_only), "Can't have both concept id only and concept values only. Please set only one of them to True."


def generate_interventions(dataset, cnt, example_idx, intervention_model, args):
    print(f"STARTING INTERVENTION GENERATION for example {example_idx} ({cnt} out of {len(args.example_idxs)})\n\n")
    init_time = time.time()
    # sub directory within output directory for this example
    example_dir = os.path.join(args.output_dir, f"example_{example_idx}")
    # init intervention generator
    ig = InterventionGenerator(
        dataset, 
        example_idx, 
        intervention_model,
        example_dir,
        concept_id_base_prompt_name=args.concept_id_base_prompt_name,
        concept_values_base_prompt_name=args.concept_values_base_prompt_name,
        counterfactual_gen_base_prompt_name=args.counterfactual_gen_base_prompt_name,
        n_workers=args.n_workers, 
        verbose=args.verbose, 
        debug=args.debug, 
        include_unknown_concept_values=args.include_unknown_concept_values,
        only_concept_removals=args.only_concept_removals,
        restart_from_previous=not args.fresh_start
        )
    # identify concepts (and their associated categories)
    concepts, categories = ig.identify_concepts()
    if args.verbose:
        print("Concepts: ", concepts)
        print("Categories for each factor: ", categories)
    if args.concept_id_only:
        print(f"FINISHED CONCEPT ID for example {example_idx} in {time.time() - init_time} seconds\n\n")
        return        
    # define intervention sets
    concept_settings = ig.define_intervention_sets(concepts)
    if args.verbose:
        print("Concept settings: ", concept_settings)
    if args.concept_values_only:
        print(f"FINISHED CONCEPT VALUES ID for example {example_idx} in {time.time() - init_time} seconds\n\n")
        return
    # apply interventions to generate counterfactual data
    ig.apply_interventions(concepts, concept_settings)
    print(f"FINISHED COUNTERFACTUAL GENERATION for example {example_idx} in {time.time() - init_time} seconds\n\n")


def main():
    args = parse_args()
    validate_args(args)
    print("ARGS...")
    print(args)
    # init dataset
    dataset = get_dataset(args.dataset, args.dataset_path)
    # init intervention model
    intervention_model = get_language_model(args.intervention_model, max_tokens=args.intervention_model_max_tokens, temperature=args.intervention_model_temperature)
    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if len(args.example_idxs) == 0:
        end_idx = len(dataset) if args.n_examples is None else args.example_idx_start + args.n_examples
        args.example_idxs = range(args.example_idx_start, end_idx)
    failed_idxs = dict()
    for cnt, example_idx in enumerate(args.example_idxs):
        try:
            generate_interventions(dataset, cnt + 1, example_idx, intervention_model, args)
        except Exception as e:
            print(f"ERROR: {e}")
            failed_idxs[example_idx] = str(e)
    # saved failed idxs and corresponding errors to file
    with open(os.path.join(args.output_dir, 'failed_idxs.json'), 'w') as f:
        json.dump(failed_idxs, f)


if __name__ == '__main__':
    main()