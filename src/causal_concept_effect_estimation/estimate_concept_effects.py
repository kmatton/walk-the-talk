import os

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random as jax_random
from numpyro.infer import MCMC, NUTS

from causal_concept_effect_estimation.concept_effect_utils import LogisticRegressionModel, MultiDatasetModel, get_category_sigma_results_from_samples, add_intrv_info_to_result_df, get_treatment_results_from_samples
from utils import load_intervention_information, load_original_model_responses, load_counterfactual_model_responses, apply_coarse_cat_mapping_to_df

class ConceptEffectEstimator:
    def __init__(self, dataset, example_idxs, intervention_data_path, model_response_path, seed=0, verbose=False):
        """
        Class for estimating the effects of concept interventions on model responses.
        Args:
        dataset: the dataset to use
        example_idxs: the indices of the examples to use
        intervention_data_path: the path to the intervention data
        model_response_path: the path to the model responses
        seed: the random seed to use
        verbose: whether to print verbose output
        """
        self.dataset = dataset
        self.example_idxs = example_idxs
        self.intervention_data_path = intervention_data_path
        self.model_response_path = model_response_path
        self.seed = seed
        self.verbose = verbose
        self.answer_choices = range(len(self.dataset.get_answer_choices()))

    def load_data(self, standardize_order=True):
        """
        Load intervention data and model responses for all examples
        """
        response_dfs = []
        for example_idx in self.example_idxs:
            response_df = self.load_example_data(example_idx)
            response_df["example_idx"] = example_idx
            response_dfs.append(response_df)
        full_response_df = pd.concat(response_dfs, ignore_index=True)
        if standardize_order:
            if not os.path.exists(os.path.join(self.model_response_path, "order_index.pkl")):
                raise Warning("Order index file not found. Ordering examples by default ordering.")
            # order examples as in our experiments in the paper
            # note the causal effect estimation is slightly sensitive to the order of the examples
            # we fix it here to achieve an exact replication, but it doesn't affect the signficance of the results
            order_idx = pd.read_pickle(os.path.join(self.model_response_path, "order_index.pkl"))
            full_response_df = full_response_df.set_index(["example_idx", "response_id"]).loc[order_idx].reset_index()
        # drop rows where answer is NaN
        full_response_df = full_response_df.dropna(subset=["answer"])
        # convert answer to int
        full_response_df["answer"] = full_response_df["answer"].astype(int)
        return full_response_df

    def load_example_data(self, example_idx):
        """
        Load intervention data and model responses for example
        Args:
            example_idx: index of example
        Returns:
            response_df: dataframe with original and counterfactual model responses
        """
        concepts, categories, concept_values = load_intervention_information(example_idx, self.intervention_data_path)
        original_responses_df = load_original_model_responses(self.model_response_path, self.dataset.name, example_idx)
        counterfactual_responses_df = load_counterfactual_model_responses(self.model_response_path, example_idx, concepts, concept_values, categories)
        if self.dataset.name == "medqa":
            # map from str answers to ints
            original_responses_df["answer"] = original_responses_df["answer"].apply(lambda x: self.dataset.get_answer_choices().index(x) if x in self.dataset.get_answer_choices() else np.NAN)
            counterfactual_responses_df["answer"] = counterfactual_responses_df["answer"].apply(lambda x: self.dataset.get_answer_choices().index(x) if x in self.dataset.get_answer_choices() else np.NAN)
        original_responses_df["intrv_str"] = "0" * len(concepts)
        original_responses_df["intrv_bool"] = [[False] * len(concepts) for _ in range(len(original_responses_df))]
        original_responses_df["intrv_idx"] = None
        original_responses_df["intrv_concept"] = None
        original_responses_df["original_value"] = None
        original_responses_df["new_value"] = None
        original_responses_df["intrv_name"] = "original"
        original_responses_df["is_original"] = True
        counterfactual_responses_df["is_original"] = False
        response_df = pd.concat([original_responses_df, counterfactual_responses_df], ignore_index=True)
        response_df["concepts"]  = [concepts for _ in range(len(response_df))]
        response_df["categories"] = [categories for _ in range(len(response_df))]
        response_df["concept_values"] = [concept_values for _ in range(len(response_df))]
        if self.dataset.name == "bbq" or self.dataset.name == "motivating-example":
            response_df["reference_class"] = self.dataset.data[example_idx]["unk_idx"]
            response_df["answer_choices"] = [[self.dataset.data[example_idx][f"ans{idx}" ] for idx in range(3)] for _ in range(len(response_df))]
        else:
            response_df["reference_class"] = 0
            response_df["answer_choices"] = [self.dataset.data[example_idx]["answer_choices"] for _ in range(len(response_df))]
        # map from fine to coarse categories
        response_df = apply_coarse_cat_mapping_to_df(response_df, self.dataset.name, coarse_cat_name="intrv_category")
        return response_df
    
    def fit_logistic_regression_hierarchical_bayesian(self, response_df):
        """
        Fit Hierarchical Bayesian Logistic Regression Model of concept effects on model responses
        Args:
            response_df: dataframe with original and counterfactual responses and metadata (concept categories and reference classes)
        Returns:
            samples: samples from the posterior distribution of the fitted model
            categories: list of categories
            treatments: list of treatments
            treatment_reference_classes: list of reference class for each treatment (e.g., for BBQ the "unk" answer choice)
        """
        # separate original and counterfactual responses
        response_df_for_modeling, treatments, categories, treatment_cats, treatment_reference_classes = self.prepare_response_data_for_modeling_all(response_df)
        # create data list for this modeling
        data_list = []
        for idx, _ in enumerate(treatments):
            data = response_df_for_modeling[response_df_for_modeling["treatment_idx"] == idx]
            X = jnp.array(data['X'])[:, None]
            Y = jnp.array(data['Y'])
            cat = jnp.array([treatment_cats[idx]])
            reference_class = treatment_reference_classes[idx]
            data_list.append((X, Y, cat, reference_class))
        n_categories = len(categories)
        # fit model
        inner_model = LogisticRegressionModel(self.answer_choices)
        model = MultiDatasetModel(inner_model, n_categories)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(jax_random.PRNGKey(0), data_list)
        samples = mcmc.get_samples()
        return samples, categories, treatments, treatment_reference_classes

    def get_parameter_results_from_posterior_samples(self, samples, categories, treatments, treatment_reference_classes, response_df):
        """
        Estimate the model parameter and treatment effect results from the posterior samples of the trained model.
        Args:
            samples: samples from the posterior distribution of the fitted model
            categories: list of categories
            treatments: list of treatments
            treatment_reference_classes: list of reference class for each treatment (e.g., for BBQ the "unk" answer choice)
            response_df: dataframe with original and counterfactual responses and metadata (concept categories and reference classes)
        Returns:
            category_parameter_df: dataframe with the mean and 95% confidence interval of the sigma parameter for each category
            treatment_parameter_df: dataframe with the mean and 95% confidence interval of the model parameters and causal effect of the treatment on the response for each treatment
        """
        # get mean and 95% confidence interval of sigma parameter for each category
        category_parameter_df = get_category_sigma_results_from_samples(samples, categories)
        print("got category parameter df")
        # get mean and 95% confidence interval of model parameters and causal effect of the treatment on the response for each treatment
        treatment_parameter_df = get_treatment_results_from_samples(samples, treatments, self.answer_choices, treatment_reference_classes)
        # add treatment info to treatment parameter df
        treatment_parameter_df["example_idx"] = treatment_parameter_df["treatment"].apply(lambda x: int(x.split("_")[0]))
        treatment_parameter_df["intrv_str"] = treatment_parameter_df["treatment"].apply(lambda x: x.split("_")[1])
        # add other intervention info
        example_df_list = []
        for example_idx in treatment_parameter_df["example_idx"].unique():
            example_result_df = treatment_parameter_df[treatment_parameter_df["example_idx"] == example_idx]
            example_response_df = response_df[response_df["example_idx"] == example_idx]
            example_result_df = add_intrv_info_to_result_df(example_result_df, example_response_df["concepts"].iloc[0], example_response_df["concept_values"].iloc[0], example_response_df["categories"].iloc[0])
            example_result_df["answer_choices"] = [example_response_df["answer_choices"].iloc[0] for _ in range(len(example_result_df))]
            intrv_ranking_idx = example_result_df.sort_values("kl_div", ascending=False).index
            example_result_df.loc[intrv_ranking_idx, "intrv_ranking"] = range(1, len(intrv_ranking_idx) + 1)
            example_df_list.append(example_result_df)
        treatment_parameter_df = pd.concat(example_df_list, ignore_index=True)
        # map from fine to coarse categories for treatment parameter df
        treatment_parameter_df = apply_coarse_cat_mapping_to_df(treatment_parameter_df, self.dataset.name, coarse_cat_name="intrv_category")
        return category_parameter_df, treatment_parameter_df

    def prepare_response_data_for_modeling_all(self, response_df):
        """
        Prepare data for Hierarchical Bayesian Modeling
        Args:
            response_df: dataframe with original and counterfactual responses and metadata (concept categories and reference classes)
        Returns:
            modeling_df: dataframe with data for Hierarchical Bayesian Modeling
            treatments: list of treatments
            categories: list of categories
            treatment_cats: list of categories for each treatment
            treatment_reference_classes: list of reference class for each treatment (e.g., for BBQ the "unk" answer choice)
        """
        # separate into original and counterfactual responses
        original_response_df = response_df[response_df["is_original"]]
        counterfactual_response_df = response_df[~response_df["is_original"]]
        
        # get full set of treatments
        counterfactual_response_df["treatment_id"] = counterfactual_response_df.apply(lambda x: f"{x['example_idx']}_{x['intrv_str']}", axis=1)
        treatments = list(counterfactual_response_df["treatment_id"].unique())
        counterfactual_response_df["treatment_idx"] = counterfactual_response_df.apply(lambda x: treatments.index(x["treatment_id"]), axis=1)
        
        # get full set of categories
        categories = list(counterfactual_response_df["intrv_category"].unique())
        counterfactual_response_df["category_idx"] = counterfactual_response_df.apply(lambda x: categories.index(x["intrv_category"]), axis=1)
        
        # get categories for each treatment
        treatment_cats = []
        # get reference classes for each treatment
        treatment_reference_classes = []
        for treatment_id in treatments:
            treatment_df = counterfactual_response_df[counterfactual_response_df["treatment_id"] == treatment_id]
            assert len(treatment_df["category_idx"].unique()) == 1, "Each treatment should have only one category."
            treatment_cats.append(treatment_df["category_idx"].iloc[0])
            treatment_reference_classes.append(treatment_df["reference_class"].iloc[0])
        
        # loop over examples
        intrv_data_list = []
        for example_idx in response_df["example_idx"].unique():
            ex_original_response_df = original_response_df[original_response_df["example_idx"] == example_idx]
            ex_counterfactual_response_df = counterfactual_response_df[counterfactual_response_df["example_idx"] == example_idx]
            
            # loop over interventions
            for intrv_str in ex_counterfactual_response_df["intrv_str"].unique():
                intrv_response_df = ex_counterfactual_response_df[ex_counterfactual_response_df["intrv_str"] == intrv_str]
                treatment_idx = treatments.index(f"{example_idx}_{intrv_str}")
                category_idx = categories.index(intrv_response_df["intrv_category"].iloc[0])
                data = pd.DataFrame({
                    'X': np.array([0] * len(ex_original_response_df) + [1] * len(intrv_response_df)),
                    'Y': ex_original_response_df["answer"].tolist() + intrv_response_df["answer"].tolist(),
                    'treatment_idx': [treatment_idx] * len(ex_original_response_df) + [treatment_idx] * len(intrv_response_df),
                    'category_idx': [category_idx] * len(ex_original_response_df) + [category_idx] * len(intrv_response_df),
                    'example_idx': [example_idx] * len(ex_original_response_df) + [example_idx] * len(intrv_response_df)
                })
                # check if any answer choices are completely missing from Y
                # if so, add one sample for each treatment to "smooth" things out
                for answer_choice in self.answer_choices:
                    if answer_choice not in data['Y'].unique():
                        smoothing_data = pd.DataFrame({
                            'X': [0, 1],
                            'Y': [answer_choice, answer_choice],
                            'treatment_idx': [treatment_idx, treatment_idx],
                            'category_idx': [category_idx, category_idx],
                            'example_idx': [example_idx, example_idx]
                        })
                        data = pd.concat([data, smoothing_data], ignore_index=True)
                intrv_data_list.append(data)
        modeling_df = pd.concat(intrv_data_list, ignore_index=True)
        return modeling_df, treatments, categories, treatment_cats, treatment_reference_classes

