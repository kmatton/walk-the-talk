# Class for determining which concepts in a question a LLM model used in making it's decision, as implied by its explanation/CoT

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import traceback
import os

import numpy as np

from IPython import embed

from utils import parse_llm_response_implied_concepts


class ExplanationAnalyzer:
    def __init__(
            self, 
            dataset,
            example_idx,
            implied_concepts_model,
            implied_concepts_base_prompt_name, 
            intervention_data_path, 
            model_response_path, 
            output_dir,
            n_completions=3,
            seed=0, 
            n_workers=4, 
            verbose=False, 
            debug=False, 
            restart_from_previous=True
        ):
        """
        Class for analyzing CoT explanations to determine which pieces in the question were used vs not.
        Args:
            dataset: dataset to use
            example_idx: index of example to analyze
            implied_concepts_model: language model to use for determining implied concepts
            implied_concepts_base_prompt_name: name of base prompt for implied concepts determination step
            intervention_data_path: path to directory with intervention data (including concepts identified and intervened on)
            model_response_path: path to directory with model responses to intervened examples
            output_dir: path to save results to
            n_completions: number of completions to generate for language model used to determine implied concepts
            seed: random seed
            n_workers: number of workers to use for parallel processing
            verbose: whether to print progress
            debug: whether to run in debug mode
            restart_from_previous: whether to restart from previous results (if they exist)
        """
        self.dataset = dataset
        self.example_idx = example_idx
        self.implied_concepts_model = implied_concepts_model
        self.implied_concepts_base_prompt_name = implied_concepts_base_prompt_name
        self.intervention_data_path = intervention_data_path
        self.model_response_path = model_response_path
        self.seed = seed
        # set seed
        np.random.seed(self.seed)
        self.n_workers = n_workers
        self.verbose = verbose
        self.debug = debug
        self.output_dir = output_dir
        self.n_completions = n_completions
        # create directory for saving results
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.restart_from_previous = restart_from_previous
        self.failures = dict()  # dict to store instances that failed

    def identify_concepts_implied_by_explanation(self, sub_dir):
        """
        Analyzes implied concepts for all responses in the model response directory for a given example
        Args:
            sub_dir: subdirectory of responses to analyze (either 'original' or 'counterfactual')
        """
        # create subdirectory for saving results
        if not os.path.exists(os.path.join(self.output_dir, sub_dir)):
            os.makedirs(os.path.join(self.output_dir, sub_dir))
        # get concepts
        concept_path = os.path.join(self.intervention_data_path, f"example_{self.example_idx}", "concepts.json")
        assert os.path.exists(concept_path), f"Missing concepts for example idx {self.example_idx} at path {concept_path}"
        with open(concept_path, 'r') as f:
            concepts = json.load(f)
        # get concept values
        concept_values_path = os.path.join(self.intervention_data_path, f"example_{self.example_idx}", "concept_settings.json")
        assert os.path.exists(concept_values_path), f"Missing concept values for example idx {self.example_idx} at path {concept_values_path}"
        with open(concept_values_path, 'r') as f:
            concept_values = json.load(f)
        response_dict = dict()
        response_dir = os.path.join(self.model_response_path, f"example_{self.example_idx}", sub_dir)
        # loop through all model responses in the model response directory for this example
        for response_file in os.listdir(response_dir):
            if not response_file.startswith("response_"):
                continue
            # check if we've already collected implied concept determination for this response
            if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, sub_dir, f"implied_concepts_{response_file}")):
                print(f"Already collected implied concepts for example {self.example_idx}, response {response_file}. Skipping...")
                continue
            with open(os.path.join(response_dir, response_file), 'r') as f:
                response = json.load(f)
            # get response
            response_dict[response_file] = (response["response"], response["answer"])
        if self.debug:
            print(f"DEBUG MODE: subsampling {len(response_dict)} responses to 5 (if 5 is smaller than current length).")
            response_dict = dict(list(response_dict.items())[:min(len(response_dict), 5)])
        # get concepts implied by the LLM's explanation for each example
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_fd_results = [executor.submit(self.identify_concepts_implied_by_model_explanation_single_example, response_id, sub_dir, response_dict, concepts, concept_values) for response_id in response_dict.keys()]
            for cnt, fd_result in enumerate(as_completed(future_fd_results)):
                fd_result.result(timeout=300)
                if cnt % 100 == 0:
                    if self.verbose:
                        print(f"Collected implied concepts for {cnt + 1} / {len(response_dict)} response for example {self.example_idx}.")

    def identify_concepts_implied_by_model_explanation_single_example(self, response_id, sub_dir, response_dict, concepts, concept_values):
        """
        Identifies concepts implied by the LLM's explanation for a single example
        Args:
            response_id: id of response to collect implied concepts for (i.e., <intervention str>_<response index>)
            sub_dir: subdirectory of responses to analyze (either 'original' or 'counterfactual')
            prompt_dict: dict mapping intervention strings to the corresponding intervened prompts
            response_dict: dict mapping response ids to the model response text and final answer
            concepts: list of concepts present in the prompt
            concept_values: list of dictionaries with the values for each concept
        """
        concepts_to_check = concepts
        values_concepts_to_check = concept_values
        if "original" in sub_dir:
            basic_prompt = self.dataset.format_prompt_basic(self.example_idx, double_space=False)
        else:
            assert sub_dir == "counterfactual", f"Invalid subdirectory {sub_dir}"
            intervention = response_id.split("_")[1].split("=")[1]
            counterfactual_file = os.path.join(self.intervention_data_path, f"example_{self.example_idx}", f"counterfactual_{intervention}.json")    
            with open(counterfactual_file, 'r') as f:
                counterfactual_dict = json.load(f)
            basic_prompt = self.dataset.format_question_counterfactual(counterfactual_dict["parsed_counterfactual"], double_space=False)
            # filter concepts to those not intervened on to set to unknown
            is_not_unknown = [x != "UNKNOWN" for x in counterfactual_dict["new_values"]]
            concepts_to_check = [f for f, i in zip(concepts, is_not_unknown) if i]
            values_concepts_to_check = [v for v, i in zip(concept_values, is_not_unknown) if i]
        model_response, model_answer = response_dict[response_id]
        prompt = self.dataset.format_prompt_implied_concepts(self.implied_concepts_base_prompt_name, concepts_to_check, values_concepts_to_check, basic_prompt, model_response, model_answer)
        try:
            responses = self.implied_concepts_model.generate_response(prompt, n_completions=self.n_completions)
            response_list = []
            concept_decision_list = []
            for r in responses:
                concept_decision, parsed_response = parse_llm_response_implied_concepts(r, len(concepts_to_check))
                response_list.append(parsed_response)
                concept_decision_list.append(concept_decision)
            fd_dict = {"prompt": prompt, "concept_decisions": concept_decision_list, "responses": response_list}
            # save concept decisions to file
            with open(os.path.join(self.output_dir, sub_dir, f"implied_concepts_{response_id}"), 'w') as f:
                json.dump(fd_dict, f)
        except Exception as e:
            print(traceback.format_exc())
            print(f"Identifying concepts implied by LLM explanation failed for example {self.example_idx}, response {response_id}. Error: {e}")
            if response_id not in self.failures:
                self.failures[response_id] = str(e)
    