import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


####################################################################################################
## Hierarchical Bayesian Modeling ##
####################################################################################################

class LinearRegressionOuterModel:
    def __init__(self, inner_model, n_feats):
        self.inner_model = inner_model
        self.n_feats = n_feats
    
    def __call__(self, data_list):
        # Prior for the regression coefficients
        mu_beta = numpyro.sample('mu_beta', dist.Normal(jnp.zeros(self.n_feats), jnp.ones(self.n_feats)))
        # Distribution for the observation noise
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        for e_idx, X, Y, _ in data_list:
            self.inner_model(X, Y, e_idx, mu_beta, sigma)

class LinearRegressionInnerModel:
    def __call__(self, X, Y, e_idx, mu_beta, sigma):
        _, n_feats = X.shape
        # Distributions for the regression coefficients
        beta = numpyro.sample(f'beta_{e_idx}', dist.Normal(mu_beta, jnp.ones(n_feats)))
        # Linear model
        preds = jnp.dot(X, beta)
        Y = numpyro.sample(f'Y_{e_idx}', dist.Normal(preds, sigma), obs=Y)


####################################################################################################
## Results Plotting ##
####################################################################################################

def plot_regression(concept_categories, concept_to_idx_map, x_vals, x, y, y_mean, y_hpdi, beta, intercept=None, cmap_idx_start=1, use_sns_palette=True, cat_legend=True, ax=None, plot_ci=True, alpha=1, legend_loc='upper left', linestyle='-', plot_faithful_line=True, line_legend=True, x_lim=[-2.5, 2.5], y_lim=[-2.5, 2.5], bbox_to_anchor=None, point_size=40):
    reverse_concept_map = {v: k for k, v in concept_to_idx_map.items()}
    # Sort values for plotting by x axis
    idx = jnp.argsort(x)
    causal_effect = x[idx]
    implied_effect = y[idx]
    sorted_categories = np.array(concept_categories)[idx]
    unique_categories = np.arange(len(concept_to_idx_map))
    if use_sns_palette:
        cmap = sns.color_palette(as_cmap=True)
        colors = cmap[cmap_idx_start:cmap_idx_start+len(unique_categories)]
    else:
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    # Plot
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    if plot_ci:
        ax.fill_between(x_vals, y_hpdi[0], y_hpdi[1], alpha=0.3, interpolate=True)
    for category in unique_categories:
        category_idx = sorted_categories == category
        label=None
        if cat_legend:
            label = reverse_concept_map[category].capitalize()
        ax.scatter(causal_effect[category_idx], implied_effect[category_idx], 
                   color=category_to_color[category], label=label, alpha=0.7, s=point_size)
    legend_str = r'$\tilde{\mathbf{EE}}$' + f' = {beta:.2f} ' + r'$\times$' + ' ' +r'$\tilde{\mathbf{CC}}$'
    if intercept and intercept > 0:
        legend_str += f' $+ {intercept:.2f}$'
    elif intercept and intercept < 0:
        legend_str += f' ${(intercept):.2f}$'
    if not line_legend:
        legend_str = None
    ax.plot(x_vals, y_mean, linestyle, label=legend_str, alpha=alpha)
    if plot_faithful_line:
        label=None
        if line_legend:
            label = r'$\tilde{\mathbf{EE}} = \tilde{\mathbf{CC}}$'
        ax.plot(x_vals, x_vals, "--", color="black", label=label, alpha=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if cat_legend or line_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
    return ax

####################################################################################################
## Data Preparation ##
####################################################################################################

def prepare_faith_data_for_regression(faith_df, concept_to_idx_map):
    """
    Prepare data for regression of explanation implied effect on causal concept effect.
    Args:
        faith_df: dataframe with causal concept effect and explanation implied effect for each concept in each example
        concept_to_idx_map: dictionary mapping concept category to index
    Returns:
        regression_data_list: list of tuples, each containing the example index, causal concept effect array, explanation implied effect array, and concept category array
        concept_cats_list: list of concept categories
        full_X_jnpy: array of causal concept effects
        full_Y_jnpy: array of explanation implied effects
    """
    regression_data_list = []
    concept_cats_list = []
    # loop through exach example and collect causal concept effect (X) and explanation implied effect (Y) for each concept
    # also collect the concept category for each concept
    for example_idx in faith_df["example_idx"].unique():
        ex_df = faith_df[faith_df["example_idx"] == example_idx]
        X = np.array(ex_df['kl_div'])[:, None]
        Y = np.array(ex_df['p(concept_in_explanation)'])
        concept_cats = [concept_to_idx_map[cat] for cat in ex_df['intrv_category'].values.tolist()]
        concept_cats_list += concept_cats
        # apply z-normalization to X and Y
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        regression_data_list.append((example_idx, jnp.array(X), jnp.array(Y), jnp.array(concept_cats)))

    full_X_jnpy = jnp.concatenate([x[1] for x in regression_data_list])
    full_Y_jnpy = jnp.concatenate([x[2] for x in regression_data_list])
    return regression_data_list, concept_cats_list, full_X_jnpy, full_Y_jnpy