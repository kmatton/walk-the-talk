from jax import random as jax_random
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from faithfulness_estimation.faithfulness_utils import prepare_faith_data_for_regression, LinearRegressionOuterModel, LinearRegressionInnerModel, plot_regression


class FaithfulnessEstimator:
    def __init__(self, ee_df, ce_df, multi_intrv_per_concept=True, categories=None):
        self.ee_df = ee_df
        self.ce_df = ce_df
        self.faith_df = ce_df.merge(ee_df, on=["example_idx", "intrv_concept", "intrv_category"])
        if multi_intrv_per_concept:
            # take mean causal concept effect for each concept in each example (across intervention settings)
            self.grouped_faith_df = self.faith_df.groupby(["example_idx", "intrv_concept"])["kl_div"].mean().to_frame().reset_index()
            self.grouped_faith_df = self.grouped_faith_df.merge(self.faith_df[["example_idx", "intrv_concept", "intrv_category", "p(concept_in_explanation)"]], on=["example_idx", "intrv_concept"], how="left").drop_duplicates()
        else:
            self.grouped_faith_df = self.faith_df
        self.categories = categories
        if categories is None:
            categories = sorted(self.grouped_faith_df['intrv_category'].unique())
        self.concept_to_idx_map = {concept: idx for idx, concept in enumerate(categories)}
        # prepare data for regression
        self.regression_data_list, self.concept_cats_list, self.full_X_jnpy, self.full_Y_jnpy = prepare_faith_data_for_regression(self.grouped_faith_df, self.concept_to_idx_map)
        self.idx_to_concept_map = {v: k for k, v in self.concept_to_idx_map.items()}

    def estimate_faithfulness(self, seed=3):
        # fit hierarchical Bayesian model
        inner_model = LinearRegressionInnerModel()
        model = LinearRegressionOuterModel(inner_model, n_feats=1)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=2000)
        mcmc.run(jax_random.PRNGKey(seed), self.regression_data_list)
        samples = mcmc.get_samples()
        # get posterior mean of beta
        beta_mean = np.mean(samples['mu_beta'])
        # get 90% credible interval of beta
        beta_credible_interval = numpyro.diagnostics.hpdi(samples['mu_beta'], 0.90)
        return samples, beta_mean, beta_credible_interval

    def plot_faithfulness(self, samples, x_min=-2.5, x_max=2.5, y_min=-2.5, y_max=2.5, keep_concepts=None):
        # compute empirical posterior predictive distribution
        x_vals = np.linspace(x_min, x_max, 100)
        posterior_preds = samples["mu_beta"] * x_vals
        mean_preds = jnp.mean(posterior_preds, axis=0)
        hpdi_preds = numpyro.diagnostics.hpdi(posterior_preds, 0.9)
        if keep_concepts is not None:
            keep_idx = np.array([True if self.idx_to_concept_map[idx] in keep_concepts else False for idx in self.concept_cats_list])
            concept_cats_list = np.array(self.concept_cats_list)[keep_idx]
            concept_to_idx_map = {v: i for i, v in enumerate(keep_concepts)}
            full_X_jnpy = self.full_X_jnpy[keep_idx]
            full_Y_jnpy = self.full_Y_jnpy[keep_idx]
        else:
            concept_cats_list = self.concept_cats_list
            concept_to_idx_map = self.concept_to_idx_map
            full_X_jnpy = self.full_X_jnpy
            full_Y_jnpy = self.full_Y_jnpy
        ax = plot_regression(concept_cats_list, concept_to_idx_map, x_vals, full_X_jnpy.flatten(), full_Y_jnpy, mean_preds, hpdi_preds, jnp.mean(samples["mu_beta"]), x_lim=[x_min, x_max], y_lim=[y_min, y_max])
        ax.set(
        xlabel=r'$\tilde{\mathbf{CE}}$ - Causal Concept Effect (Z-Score)', ylabel=r'$\tilde{\mathbf{EE}}$ - Explanation Implied Effect (Z-Score)'
        )
        