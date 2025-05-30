# Class for generating interventional data for a single example

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import threading
import traceback

from IPython import embed

from utils import parse_llm_response_concepts_and_categories, parse_llm_response_factor_settings, enumerate_interventions


class InterventionGenerator:
    def __init__(self, dataset, example_idx, intervention_model, output_dir, concept_id_base_prompt_name, concept_values_base_prompt_name, counterfactual_gen_base_prompt_name, n_workers=4, seed=42, verbose=False, debug=False,
                 include_unknown_concept_values=True, only_concept_removals=False, restart_from_previous=True):
        """
        Args:
            dataset: dataset to use
            example_idx: index of example to generate interventions on
            intervention_model: language model to use to generate counterfactual data
            output_dir: directory to save results to
            concept_id_base_prompt_name: name of the concept ID base prompt to use
            concept_values_base_prompt_name: name of the concept values base prompt to use
            counterfactual_gen_base_prompt_name: name of the counterfactual generation base prompt to use
            n_workers: number of workers to use for parallelization
            seed: random seed
            verbose: whether to print progress
            debug: whether to run in debug mode
            include_unknown_concept_values: whether to include 'unknown' as a concept value in the intervention generation
            only_concept_removals: whether to only generate interventions that remove concepts
            restart_from_previous: if True, restart from progress made on previous run
        """
        self.dataset = dataset
        self.example_idx = example_idx
        self.intervention_model = intervention_model
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(output_dir)
        self.concept_id_base_prompt_name = concept_id_base_prompt_name
        self.concept_values_base_prompt_name = concept_values_base_prompt_name
        self.counterfactual_gen_base_prompt_name = counterfactual_gen_base_prompt_name
        self.n_workers = n_workers
        self.verbose = verbose
        self.debug = debug
        self.seed = seed
        self.include_unknown_concept_values = include_unknown_concept_values
        self.only_concept_removals = only_concept_removals
        self.restart_from_previous = restart_from_previous
        # set seed
        random.seed(self.seed)

    def identify_concepts(self):
        """
        Identifies concepts to test for a given question.
        Args:
            dataset: dataset object
        Returns:
            concepts: a parsed list of concepts identified by the LLM
            categories: a parsed list of categories associated with each concept
        """
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concepts.json')):
            print("Found existing concepts.json file. Skipping concept identification...")
            with open(os.path.join(self.output_dir, 'concepts.json'), 'r') as f:
                concepts = json.load(f)
            assert os.path.exists(os.path.join(self.output_dir, 'categories.json')), "categories.json file not found"
            with open(os.path.join(self.output_dir, 'categories.json'), 'r') as f:
                categories = json.load(f)
            return concepts, categories
        # get concept ID prompt
        prompt = self.dataset.format_prompt_concept_id(self.example_idx, self.concept_id_base_prompt_name)
        # query model for concepts
        response = self.intervention_model.generate_response(prompt)[0]
        # parse response for concepts
        try:
            concepts, categories = parse_llm_response_concepts_and_categories(response)
            # save concept results to file
            with open(os.path.join(self.output_dir, 'concepts.json'), 'w') as f:
                json.dump(concepts, f)
            with open(os.path.join(self.output_dir, 'categories.json'), 'w') as f:
                json.dump(categories, f)
            return concepts, categories
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Concept identification failed: {e}")
    
    def define_intervention_sets(self, concepts):
        """
        Defines intervention sets for a given question (i.e., the values to set when intervening on a concept).
        Args:
            concepts: list of the concepts identified by the LLM
        Returns:
            concept_settings: a list of settings for each concept
        """
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concept_settings.json')):
            print("Found existing concept_settings.json file. Skipping concept settings identification...")
            with open(os.path.join(self.output_dir, 'concept_settings.json'), 'r') as f:
                concept_settings = json.load(f)
            return concept_settings
        # get concept settings prompt
        prompt = self.dataset.format_prompt_concept_values(self.example_idx, self.concept_values_base_prompt_name, concepts)
        # query model for concept settings
        response = self.intervention_model.generate_response(prompt)[0]
        # parse response for concept settings
        try:
            concept_settings = parse_llm_response_factor_settings(response)
            # save concept settings
            with open(os.path.join(self.output_dir, 'concept_settings.json'), 'w') as f:
                json.dump(concept_settings, f)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Concept settings identification failed: {e}")
        return concept_settings
    
    def apply_interventions(self, concepts, concept_settings):
        """
        Apply interventions for a given example.
        Args:
            concepts: the concepts to intervene on
            concept_settings: the settings for each factor
        """
        # check for existing interventions
        existing_interventions = []
        if self.restart_from_previous:
            existing_interventions = [x.split('.')[0].split('_')[1] for x in os.listdir(self.output_dir) if x.startswith('counterfactual_')]
            if len(existing_interventions) > 0:
                print(f"Found {len(existing_interventions)} existing interventions. Skipping these...")
        # if only doing concept removals, remove new settings from factor settings
        if self.only_concept_removals:
            for factor_setting in concept_settings:
                factor_setting["new_settings"] = ["UNKNOWN"]
        # if unknown concept values are included, add them to the factor settings
        if self.include_unknown_concept_values and not self.only_concept_removals:
            for factor_setting in concept_settings:
                factor_setting["new_settings"].append("UNKNOWN")
        # generate interventions by intervening on each concept to set to each of the possible settings
        intervention_list = enumerate_interventions(concepts, concept_settings, k_hop=1, include_no_intervention=False, mark_removals=True)
        # remove existing interventions from this list
        intervention_list = [x for x in intervention_list if x not in existing_interventions]
        if self.debug and len(intervention_list) >= 10: # sample subset of interventions
            intervention_list = intervention_list[:10]
            if self.verbose:
                print(f"DEBUG mode: executing 10 interventions out of {len(intervention_list)}...")
        if len(intervention_list) == 0: 
            print("No interventions to apply. Exiting...")
            return
        print(f"Executing {len(intervention_list)} interventions...")
        interventions = []
        print(f"Using {self.n_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.n_workers) as executer:
            future_intrvs = [executer.submit(self.apply_single_intervention, i_str, concepts, concept_settings) for i_str in intervention_list]
            for cnt, i_result in enumerate(as_completed(future_intrvs)):
                i_result_dict = i_result.result(timeout=300)
                interventions.append(i_result_dict)
                if self.verbose and cnt % 100 == 0:
                    print(f"Finished {cnt+1}/{len(intervention_list)} interventions")
                    print(f"Threading active count {threading.active_count()}")

    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """
        Apply an intervention to a given example, and extract the model's response.
        Save the intervened example to a json file.
        Args:
            intervention_str: string specifying the intervention settings
            concepts: the concepts to intervene on
            concept_settings: the settings for each concept
        """
        # identify settings of concepts that are intervened on
        old_values = [x["current_setting"] for x in concept_settings]
        new_values = copy.deepcopy(old_values)
        for idx, val in enumerate(intervention_str):
            if val == '-':
                new_values[idx] = "UNKNOWN"
            else:
                val_int = int(val)
                if val_int:
                    new_values[idx] = concept_settings[idx]["new_settings"][val_int - 1]
        # generate intervened prompt
        intervention_bool = [True if intervention_str[i] != "0" else False for i in range(len(intervention_str))]
        counterfactual_gen_prompt = self.dataset.format_prompt_counterfactual_gen(
            self.example_idx, 
            self.counterfactual_gen_base_prompt_name, 
            concepts, intervention_bool, 
            new_values, 
            old_values
            )
        counterfactual = self.intervention_model.generate_response(counterfactual_gen_prompt)[0].strip()
        try:
            parsed_counterfactual = self.dataset.parse_counterfactual_output(counterfactual)
        except Exception as e:
            embed()
            print(traceback.format_exc())
            raise Exception(f"Parsing failed for counterfactual {counterfactual}. Error was: {e}")
        intervention_dict = {
            "intervention_str": intervention_str,
            "old_values": old_values,
            "new_values": new_values,
            "counterfactual": counterfactual,
            "counterfactual_gen_prompt": counterfactual_gen_prompt,
            "parsed_counterfactual": parsed_counterfactual
        }
        with open(os.path.join(self.output_dir, f'counterfactual_{intervention_str}.json'), 'w') as f:
            json.dump(intervention_dict, f)
