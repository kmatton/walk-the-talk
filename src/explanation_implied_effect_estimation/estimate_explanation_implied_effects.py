import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from utils import process_intervention_str, load_intervention_information, apply_coarse_cat_mapping_to_df


class ExplanationImpliedEffectEstimator:
    def __init__(self, dataset, example_idxs, intervention_data_path, implied_concepts_path, seed=0, verbose=False):
        """
        Class for estimating explanation implied effects for a given dataset and set of examples.
        Args:
            dataset: dataset to use
            example_idxs: list of example indices to use
            intervention_data_path: path to intervention data
            implied_concepts_path: path to directory with implied concepts outputs from auxiliary LLM
            seed: random seed
            verbose: whether to print progress
        """
        self.dataset = dataset
        self.example_idxs = example_idxs
        self.intervention_data_path = intervention_data_path
        self.implied_concepts_path = implied_concepts_path
        self.seed = seed
        self.verbose = verbose
        self.answer_choices = range(len(self.dataset.get_answer_choices()))

    def load_data(self, load_counterfactual_responses=True):
        """
        Load implied concepts determinations for LLM responses
        Args:
            load_counterfactual_responses: whether to load analysis for counterfactual responses (in addition to original responses)
        Returns:
            list of dataframes with implied concepts determinations for each LLM response
        """
        response_dfs = []
        for example_idx in self.example_idxs:
            response_df = self.load_example_data(example_idx, load_counterfactual_responses=load_counterfactual_responses)
            response_df["example_idx"] = example_idx
            response_dfs.append(response_df)
        return pd.concat(response_dfs, ignore_index=True)
    
    def estimate_implied_effects(self, ic_df):
        """
        Estimate explanation implied effects for a given dataframe with implied concepts determinations
        Args:
            ic_df: dataframe with implied concepts determinations
        Returns:
            explanation_implied_effects_df: dataframe with explanation implied effects (in the p(concept_in_explanation) column)
        """
        concept_scores_dict = {
        "example_idx": [],
        "intrv_concept": [],
        "intrv_category": [],
        "p(concept_in_explanation)": [],
        "concept_ranking": [],
        }
        for example_idx in self.example_idxs:
            ex_df = ic_df[(ic_df["example_idx"] == example_idx)]
            ex_concept_decisions = [x[0] for x in ex_df["concept_decisions"].values]
            concept_scores_dict["intrv_concept"] += ex_df["concepts"].values[0]
            concept_scores_dict["intrv_category"] += ex_df["categories"].values[0]
            fd_means = list(np.mean(ex_concept_decisions, axis=0))
            concept_scores_dict["p(concept_in_explanation)"] += fd_means
            concept_scores_dict["example_idx"] += [example_idx] * len(fd_means)
            concept_scores_dict["concept_ranking"] += list(rankdata(-1 * np.array(fd_means), method="min"))
        explanation_implied_effects_df = pd.DataFrame(concept_scores_dict)
        explanation_implied_effects_df = apply_coarse_cat_mapping_to_df(explanation_implied_effects_df, self.dataset.name, coarse_cat_name="intrv_category")
        return explanation_implied_effects_df
    
    def load_example_data(self, example_idx, load_counterfactual_responses=True):
        """
        Load intervention data and implied concepts for a single example
        Args:
            example_idx: index of example
        Returns:
            dataframe with implied concepts determinations
        """
        concepts, categories, concept_values = load_intervention_information(example_idx, self.intervention_data_path)
        original_ic_df = self.load_original_implied_concept_determinations(example_idx)
        original_ic_df["intrv_str"] = "0" * len(concepts)
        original_ic_df["intrv_bool"] = [[False] * len(concepts) for _ in range(len(original_ic_df))]
        original_ic_df["intrv_idx"] = None
        original_ic_df["intrv_concept"] = None
        original_ic_df["original_value"] = None
        original_ic_df["new_value"] = None
        original_ic_df["intrv_name"] = "original"
        original_ic_df["is_original"] = True
        ic_df = original_ic_df
        if load_counterfactual_responses:
            counterfactual_ic_df = self.load_counterfactual_implied_concept_determinations(example_idx, concepts, concept_values, categories)
            counterfactual_ic_df["is_original"] = False
            ic_df = pd.concat([original_ic_df, counterfactual_ic_df], ignore_index=True)
        ic_df["concepts"]  = [concepts for _ in range(len(ic_df))]
        ic_df["categories"] = [categories for _ in range(len(ic_df))]
        ic_df["concept_values"] = [concept_values for _ in range(len(ic_df))]
        return ic_df
    
    def load_original_implied_concept_determinations(self, example_idx):
        """
        Load implied concepts determinations for original model responses for a given example
        Args:
            example_idx: index of example
        Returns:
            dataframe with implied concepts determinations for original model responses
        """
        ic_dict = {"response_id": [], "prompt": [], "responses": [], "concept_decisions": []}
        example_original_response_dir = os.path.join(self.implied_concepts_path, f"example_{example_idx}", "original")
        for response_file in os.listdir(example_original_response_dir):
            assert response_file.startswith("implied_concepts_response_n="), f"Invalid response file: {response_file}"
            with open(os.path.join(example_original_response_dir, response_file), "r") as f:
                response = json.load(f)
            response_id = int(response_file.split("implied_concepts_response_n=")[1].split(".json")[0])
            ic_dict["response_id"].append(f"original_n={response_id}")
            ic_dict["prompt"].append(response["prompt"])
            ic_dict["responses"].append(response["responses"])
            ic_dict["concept_decisions"].append(response["concept_decisions"])
        return pd.DataFrame(ic_dict)
    
    def load_counterfactual_implied_concept_determinations(self, example_idx, concepts, concept_values, categories):
        """
        Load implied concepts determinations for counterfactual model responses for a given example
        Args:
            example_idx: index of example
            concepts: list of concepts that interventions were applied to
            concept_values: list of concept values
            categories: list of categories associated with each concept
        Returns:
            dataframe with implied concepts determinations for counterfactual model responses
        """
        response_dict = {"intrv_str": [], 
                         "intrv_bool": [],
                         "intrv_idx": [],
                         "intrv_concept": [],
                         "intrv_category": [],
                         "original_value": [],
                         "new_value": [],
                         "intrv_name": [],
                         "response_id": [], 
                         "prompt": [], 
                         "responses": [], 
                         "concept_decisions": [], 
                         }
        example_counterfactual_response_dir = os.path.join(self.implied_concepts_path, f"example_{example_idx}", "counterfactual")
        for response_file in os.listdir(example_counterfactual_response_dir):
            assert response_file.startswith("implied_concepts_response_counterfactual="), f"Invalid response file: {response_file}"
            with open(os.path.join(example_counterfactual_response_dir, response_file), "r") as f:
                response = json.load(f)
            intervention_str = response_file.split("implied_concepts_response_counterfactual=")[1].split("_n=")[0]
            completion_id = int(response_file.split("implied_concepts_response_counterfactual=")[1].split("_n=")[1].split(".json")[0])
            response_dict["intrv_str"].append(intervention_str)
            response_dict["response_id"].append(f"counterfactual={intervention_str}_n={completion_id}")
            response_dict["prompt"].append(response["prompt"])
            response_dict["responses"].append(response["responses"])
            intrv_bool, intrv_idx, intrv_concept, intrv_category, original_value, new_value, intrv_name = process_intervention_str(intervention_str, concepts, concept_values, categories)
            response_dict["intrv_bool"].append(intrv_bool)
            response_dict["intrv_idx"].append(intrv_idx)
            response_dict["intrv_concept"].append(intrv_concept)
            response_dict["intrv_category"].append(intrv_category)
            response_dict["original_value"].append(original_value)
            response_dict["new_value"].append(new_value)
            response_dict["intrv_name"].append(intrv_name)
            concept_decisions = response["concept_decisions"]
            if new_value == "UNKNOWN":
                # need to add in 0 for "unknown" concept that we didn't list in options that model could mention for counterfactual questions
                for idx in range(len(concept_decisions)):
                    concept_decisions[idx].insert(intrv_idx, 0)
            response_dict["concept_decisions"].append(concept_decisions)
        return pd.DataFrame(response_dict)
 