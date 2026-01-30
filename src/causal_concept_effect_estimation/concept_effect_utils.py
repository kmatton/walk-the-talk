import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
import pandas as pd

from utils import process_intervention_str

####################################################################################################
## Hierarchical Bayesian Modeling ##
####################################################################################################

class LogisticRegressionModel:
    def __init__(self, response_categories):
        self.response_categories = response_categories
        self.n_responses = len(response_categories)
    
    def __call__(self, scale, X, Y, cat, e_idx, reference_class):
        """
        Args:
            scale: scale of beta prior
            X: features
            Y: responses
            e_idx: index of example in dataset that intervention data is associated with
        """
        n_samples, n_feats = X.shape
        # sample intercept
        intercept = numpyro.sample(f'intercept{e_idx}', dist.Normal(0, 1), sample_shape=(self.n_responses-1,))
        # sample betas for each feature, from category-specific priors
        if n_feats != 0:
            with numpyro.plate(f"feature_{e_idx}", n_feats):
                beta_scale = scale[cat]
                beta = numpyro.sample(f'beta{e_idx}', dist.Normal(0, beta_scale), sample_shape=(self.n_responses-1,))
            logits = jnp.matmul(X, jnp.transpose(beta)) + intercept
        else:
            logits = jnp.zeros((n_samples, self.n_responses-1)) + intercept
        # add in logits for reference class
        logits = jnp.insert(logits, reference_class, jnp.zeros(n_samples), axis=1)
        Y = numpyro.sample(f'Y{e_idx}', dist.Categorical(logits=logits), obs=Y)


class MultiDatasetModel:
    def __init__(self, inner_model, n_categories):
        self.inner_model = inner_model
        self.n_categories = n_categories
    
    def __call__(self, data_list):
        # put prior on variance of beta for each category
        scale = numpyro.sample('sigma', dist.InverseGamma(0.001, 0.001), sample_shape=(self.n_categories,))
        for e_idx, data in enumerate(data_list):
            X, y, cat, reference_class = data
            self.inner_model(scale, X, y, cat, e_idx, reference_class)


####################################################################################################
## Results Processing ##
####################################################################################################

def add_intrv_info_to_result_df(result_df, concepts, concept_values, categories):
    """
    Add the intervention information to the result dataframe for a single example.
    Args:
        result_df: dataframe with the results
        concepts: list of concepts for the example
        concept_values: list of concept values for each concept
        categories: list of categories associated with each concept
    Returns:
        DataFrame with the intervention information added to the result dataframe
    """
    result_df["intrv_name"] = ""
    result_df["intrv_category"] = ""
    def add_intrv_info_to_row(x):
        intrv_bool, intrv_idx, intrv_concept, intrv_category, original_value, new_value, intrv_name = process_intervention_str(x["intrv_str"], concepts, concept_values, categories)
        x["intrv_bool"] = intrv_bool
        x["intrv_idx"] = intrv_idx
        x["intrv_concept"] = intrv_concept
        x["intrv_category"] = intrv_category
        x["original_value"] = original_value
        x["new_value"] = new_value
        x["intrv_name"] = intrv_name
        return x
    result_df = result_df.apply(add_intrv_info_to_row, axis=1)
    return result_df

def get_category_sigma_results_from_samples(samples, categories):
    """
    Get the mean and 95% confidence interval of the sigma parameter for each category.
    Args:
        samples: samples from the model
        categories: categories of the interventions
    Returns:
        DataFrame with the mean and 95% confidence interval of the sigma parameter for each category
    """
    result_dict = {"category": categories}
    sigma_means = np.mean(samples["sigma"], axis=0)
    result_dict["sigma"] = sigma_means
    ci = numpyro.diagnostics.hpdi(samples["sigma"], 0.95, axis=0)
    result_dict["sigma_ci_low"] = ci[0]
    result_dict["sigma_ci_high"] = ci[1]
    return pd.DataFrame(result_dict)

def compute_probabilities(intercept, beta, X, reference_class):
    """
    Compute the probabilities of each response for a given intercept, beta, X, and reference class.
    Args:
        intercept: intercept of the logistic regression model
        beta: beta of the logistic regression model
        X: features of the input (0/1 for control/treatment)
        reference_class: reference class in the logistic regression model
    Returns:
        probabilities of each response
    """
    logits = intercept[:, None] + beta * jnp.array([X])
    # add in reference class logits
    logits = jnp.insert(logits, reference_class, 0, axis=0)
    exp_logits = jnp.exp(logits)
    return exp_logits / exp_logits.sum()

def get_posterior_dist_causal_effect_estimates_hierarchical(samples, reference_class, treatment_idx):
    """
    Approximate the posterior distribution of the causal effect of the treatment on the response using the samples from the model.
    Args:
        samples: samples from the model
        reference_class: reference class in the logistic regression model
        treatment_idx: index of the treatment
    Returns:
        Dictionary with the mean and 95% confidence interval of the causal effect of the treatment on the response, measured as KL divergence
        between the response distributions for the treatment and control group. The dictionary also includes the mean and 95% confidence interval of the answer probabilities for each class for the treatment and control group.
    """
    # initialize lists to store results for each sample
    prob_control_samples = []
    prob_treatment_samples = []
    beta_samples = []
    kl_divergence_samples = []
    n_samples = len(samples[f'intercept{treatment_idx}'])

    # loop through samples and compute (1) probabilities for control and treatment, (2) KL divergence
    for i in range(n_samples):
        intercept = samples[f'intercept{treatment_idx}'][i]
        beta = samples[f'beta{treatment_idx}'][i]
        beta_samples.append(beta)
        prob_control = compute_probabilities(intercept, beta, 0, reference_class).flatten()
        prob_control_samples.append(prob_control)
        prob_treatment = compute_probabilities(intercept, beta, 1, reference_class).flatten()
        prob_treatment_samples.append(prob_treatment)
        kl_divergence = jnp.sum(prob_treatment * jnp.log(prob_treatment / prob_control))
        kl_divergence_samples.append(kl_divergence)
    
    # compute mean and 95% confidence interval of results
    prob_control_samples = jnp.array(prob_control_samples)
    prob_treatment_samples = jnp.array(prob_treatment_samples)
    prob_control_mean = jnp.mean(prob_control_samples, axis=0)
    prob_control_hdi = numpyro.diagnostics.hpdi(prob_control_samples, 0.95, axis=0)
    prob_treatment_mean = jnp.mean(prob_treatment_samples, axis=0)
    prob_treatment_hdi = numpyro.diagnostics.hpdi(prob_treatment_samples, 0.95, axis=0)
    kl_divergence_samples = jnp.array(kl_divergence_samples)
    kl_divergence_mean = jnp.mean(kl_divergence_samples)
    kl_divergence_hdi = numpyro.diagnostics.hpdi(kl_divergence_samples, 0.95)
    return {
        'prob_control_mean': prob_control_mean,
        'prob_control_hdi': prob_control_hdi,
        'prob_treatment_mean': prob_treatment_mean,
        'prob_treatment_hdi': prob_treatment_hdi,
        'kl_div_mean': kl_divergence_mean,
        'kl_div_hdi': kl_divergence_hdi,
        'kl_div_samples': np.array(kl_divergence_samples)  
    }

def get_treatment_results_from_samples(samples, treatments, answer_choices, reference_classes):
    """
    Get the mean and 95% confidence interval of the parameters and causal effect of the treatment on the response for each treatment.
    Args:
        samples: samples from the model
        treatments: list of treatments
        answer_choices: list of the answer choice options for the response variable 
        reference_classes: reference classes for each treatment's logistic regression sub-model
    Returns:
        DataFrame with the mean and 95% confidence interval of the parameters and causal effect of the treatment on the response for each treatment
    """
    # initialize dictionary to store results
    parameter_names = ["beta", "intercept"]
    result_dict = {"treatment": treatments}
    result_dict.update({f"{parameter}_{answer_choice}": [] for parameter in parameter_names for answer_choice in answer_choices})
    result_dict.update({f"{parameter}_ci_low_{answer_choice}": [] for parameter in parameter_names for answer_choice in answer_choices})
    result_dict.update({f"{parameter}_ci_high_{answer_choice}": [] for parameter in parameter_names for answer_choice in answer_choices})
    result_cols = [f"p_y={answer_choice}|X=1" for answer_choice in answer_choices] \
            + [f"p_y={answer_choice}|X=1_ci_low" for answer_choice in answer_choices] \
            + [f"p_y={answer_choice}|X=1_ci_high" for answer_choice in answer_choices] \
            + [f"p_y={answer_choice}|X=0" for answer_choice in answer_choices] \
            + [f"p_y={answer_choice}|X=0_ci_low" for answer_choice in answer_choices] \
            + [f"p_y={answer_choice}|X=0_ci_high" for answer_choice in answer_choices] \
            + ["kl_div"] \
            + ["kl_div_ci_low"] \
            + ["kl_div_ci_high"] \
            + ['kl_div_samples']
    result_dict.update({col: [] for col in result_cols})
    
    # get results for each treatment
    for idx in range(len(treatments)):
        print(f"working on treatment {idx + 1} out of {len(treatments)}")
        reference_class_idx = reference_classes[idx]
        posterior_ce_estimates = get_posterior_dist_causal_effect_estimates_hierarchical(samples, reference_class_idx, idx)
        non_reference_class_idxs = [x for x in answer_choices if x != reference_class_idx]
        
        # get results (mean and 95% confidence interval) for each parameter
        for parameter in parameter_names:
            parameter_means = np.mean(samples[f"{parameter}{idx}"], axis=0).flatten()
            parameter_ci = numpyro.diagnostics.hpdi(samples[f"{parameter}{idx}"], 0.95, axis=0)
            for answer_choice in answer_choices:
                if answer_choice != reference_class_idx:
                    shifted_idx = non_reference_class_idxs.index(answer_choice)
                    result_dict[f"{parameter}_{answer_choice}"].append(parameter_means[shifted_idx])
                    result_dict[f"{parameter}_ci_low_{answer_choice}"].append(parameter_ci[0].flatten()[shifted_idx])
                    result_dict[f"{parameter}_ci_high_{answer_choice}"].append(parameter_ci[1].flatten()[shifted_idx])

                else:
                    result_dict[f"{parameter}_{answer_choice}"].append(None)
                    result_dict[f"{parameter}_ci_low_{answer_choice}"].append(None)
                    result_dict[f"{parameter}_ci_high_{answer_choice}"].append(None)

        # get general CE results
        for answer_choice in answer_choices:
            result_dict[f"p_y={answer_choice}|X=1"].append(posterior_ce_estimates['prob_treatment_mean'][answer_choice].item())
            result_dict[f"p_y={answer_choice}|X=1_ci_low"].append(posterior_ce_estimates['prob_treatment_hdi'][0][answer_choice])
            result_dict[f"p_y={answer_choice}|X=1_ci_high"].append(posterior_ce_estimates['prob_treatment_hdi'][1][answer_choice])
            result_dict[f"p_y={answer_choice}|X=0"].append(posterior_ce_estimates['prob_control_mean'][answer_choice].item())
            result_dict[f"p_y={answer_choice}|X=0_ci_low"].append(posterior_ce_estimates['prob_control_hdi'][0][answer_choice])
            result_dict[f"p_y={answer_choice}|X=0_ci_high"].append(posterior_ce_estimates['prob_control_hdi'][1][answer_choice])
        
        result_dict["kl_div"].append(posterior_ce_estimates['kl_div_mean'].item())
        result_dict["kl_div_ci_low"].append(posterior_ce_estimates['kl_div_hdi'][0])
        result_dict["kl_div_ci_high"].append(posterior_ce_estimates['kl_div_hdi'][1])
        result_dict["kl_div_samples"].append(posterior_ce_estimates['kl_div_samples'])
    return pd.DataFrame(result_dict)


####################################################################################################
## Reading in Data ##
####################################################################################################

