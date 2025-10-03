# Class for collecting response of LLM to intervened questions.

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

import numpy as np
from IPython import embed


class ResponseCollector:
    def __init__(self, dataset, example_idx, intervention_data_path, language_model, prompt_strategy, output_dir, n_completions=5, seed=0, 
                 n_workers=4, verbose=False, debug=False, restart_from_previous=True, save_failed_responses=True):
        """
        Class for collecting LLM responses to intervened/perturbed questions.
        Args:
            dataset: dataset to collect responses from
            example_idx: idx of example to collect responses for
            intervention_data_path: path to directory with intervention data
            language_model: language model to collect responses from
            prompt_strategy: prompting strategy to use when collecting responses
            output_dir: directory to save model responses to
            n_completions: number of completions to generate for each intervened example
            seed: random seed
            n_workers: number of workers to use for parallel processing
            verbose: whether to print progress
            debug: whether to run in debug mode
            restart_from_previous: whether to restart from previous results (if they exist)
            save_failed_responses: whether to save failed responses
        """
        self.dataset = dataset
        self.example_idx = example_idx
        self.intervention_data_path = intervention_data_path
        self.language_model = language_model
        self.prompt_strategy = prompt_strategy
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.n_completions = n_completions
        self.seed = seed
        # set seed
        np.random.seed(self.seed)
        self.n_workers = n_workers
        self.verbose = verbose
        self.debug = debug
        self.restart_from_previous = restart_from_previous
        self.save_failed_responses = save_failed_responses
        self.failures = dict()  # dict to store instances that fail

    def collect_original_model_responses(self):
        """
        Collects model responses for the original question
        Args:
            example_idx: idx of single base example for which to collect responses
        """
        original_output_dir = os.path.join(self.output_dir, "original")
        if not os.path.exists(original_output_dir):
            os.makedirs(original_output_dir)
        # check for existing outputs
        completed = []
        if self.restart_from_previous:
            for i in range(self.n_completions):
                if os.path.exists(os.path.join(original_output_dir, f"response_n={i}.json")):
                    completed.append(i)
            if len(completed) == self.n_completions:
                print(f"Already collected all responses to original question for example={self.example_idx}. Skipping...")
                return
        completions_to_get = [i for i in range(self.n_completions) if i not in completed]
        try:
            basic_prompt = self.dataset.format_prompt_basic(self.example_idx)
            # format prompt based on prompting strategy
            qa_prompt = self.dataset.format_prompt_qa(basic_prompt, self.prompt_strategy, idx=self.example_idx)
            # get model output on original prompt
            original_responses = self.language_model.generate_response(qa_prompt, n_completions=len(completions_to_get))
        except Exception as e:
            print(f"Failed to generate model response to original question for example {self.example_idx}: {e}")
            self.failures["original"] = completions_to_get
            return
        # get answers
        for response_idx, response in enumerate(original_responses):
            idx_name = completions_to_get[response_idx]
            try:
                answer = self.dataset.extract_answer(response, self.prompt_strategy, self.example_idx)
                answer_dict = {"prompt": qa_prompt, "response": response, "answer": answer}
                # save response
                with open(os.path.join(original_output_dir, f"response_n={idx_name}.json"), 'w') as f:
                    json.dump(answer_dict, f)
            except Exception as e:
                print(f"Failed to extract answer for original question for example {self.example_idx}.\nResponse was {response}\nResponse index: {idx_name}.\nError: {e}")
                if "original" not in self.failures:
                    self.failures["original"] = []
                self.failures["original"].append(idx_name)

    def collect_counterfactual_model_responses(self):
        """
        Collects model responses for all intervened/perturbed examples for a single base example
        """
        counterfactual_output_dir = os.path.join(self.output_dir, "counterfactual")
        if not os.path.exists(counterfactual_output_dir):
            os.makedirs(counterfactual_output_dir)
        prompt_dict = {} # dict to store intervened prompts to use
        completions_dict = {} # dict to store number of completions to use for each intervention
        # make response dir for this example if it doesn't already exist
        example_dir = f"example_{self.example_idx}"
        # loop through all intervened examples in the intervention data example directory
        for intervention_file in os.listdir(os.path.join(self.intervention_data_path, example_dir)):
            if not intervention_file.startswith("counterfactual_"):
                continue
            # get intervention_str
            intrv_str = intervention_file.split(".")[0].split("_")[1]
            # check if we already have responses for this intervention 
            completed = []
            if self.restart_from_previous:
                for i in range(self.n_completions):
                    if os.path.exists(os.path.join(counterfactual_output_dir, f"response_counterfactual={intrv_str}_n={i}.json")):
                        completed.append(i)
            if len(completed) == self.n_completions:
                print(f"Already collected responses for example={self.example_idx} counterfactual={intrv_str}. Skipping...")
                continue
            # read file
            with open(os.path.join(self.intervention_data_path, example_dir, intervention_file), "r") as f:
                intervention_data = json.load(f)
            prompt_dict[intrv_str] = intervention_data["parsed_counterfactual"]
            completions_dict[intrv_str] = [i for i in range(self.n_completions) if i not in completed]
            assert intrv_str == intervention_data["intervention_str"], "intervention_str in filename does not match intervention_str in file"
        print(f"FOUND {len(prompt_dict)} INTERVENTIONS FOR EXAMPLE {self.example_idx}.")
        interventions = list(prompt_dict.keys())
        if self.debug:
            interventions = interventions[:5]
            print(f"DEBUG MODE: subsampling {len(prompt_dict)} interventions to 5.")
        # collect responses for all interventions for this example
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self.collect_response_single_intrv, counterfactual_output_dir, intrv_str, prompt_dict, completions_dict) for intrv_str in interventions]
            for cnt, future in enumerate(as_completed(futures)):
                future.result(timeout=300)
                if cnt % 100 == 0:
                    if self.verbose:
                        print(f"Collected responses for {cnt + 1} / {len(prompt_dict)} interventions for {self.example_idx}.")

    def collect_response_single_intrv(self, output_dir, intrv_str, prompt_dict, completions_dict):
        """
        Collects model responses for a single intervention/perturbed example
        Args:
            output_dir: directory to save responses to
            intrv_str: intervention string associated with intervention to collect responses for 
            prompt_dict: dict mapping intervention strs to prompts, for a single example
            n_completions: dict mapping intervention strs to list of completions needed, for a single example
        """
        n_completions_to_extract = len(completions_dict[intrv_str])
        try:
            # format prompt based on prompting strategy
            intervention_prompt = self.dataset.format_prompt_qa_counterfactual(prompt_dict[intrv_str], self.prompt_strategy, idx=self.example_idx)
            # get model output on intervened prompt
            intervention_responses = self.language_model.generate_response(intervention_prompt, n_completions=n_completions_to_extract)
        except Exception as e:
            print(f"Failed to generate model response for example {self.example_idx}, intervention {intrv_str}: {e}")
            self.failures[intrv_str] = completions_dict[intrv_str]
            return
        # get answers
        for response_idx, response in enumerate(intervention_responses):
            idx_name = completions_dict[intrv_str][response_idx]
            try:
                answer = self.dataset.extract_answer(response, self.prompt_strategy, self.example_idx)
                answer_dict = {"prompt": intervention_prompt, "response": response, "answer": answer}
                # save response
                with open(os.path.join(output_dir, f"response_counterfactual={intrv_str}_n={idx_name}.json"), 'w') as f:
                    json.dump(answer_dict, f)
            except Exception as e:
                print(f"Failed to extract expected answer choice for example {self.example_idx}, intervention {intrv_str}.\nResponse was {response}\nResponse index: {idx_name}.\nError: {e}")
                if self.save_failed_responses:
                # save response anyway
                    answer_dict = {"prompt": intervention_prompt, "response": response, "answer": "NaN", "error": str(e)}
                    with open(os.path.join(output_dir, f"response_counterfactual={intrv_str}_n={idx_name}.json"), 'w') as f:
                        json.dump(answer_dict, f)
                if intrv_str not in self.failures:
                    self.failures[intrv_str] = []
                self.failures[intrv_str].append({idx_name: str(e)})
