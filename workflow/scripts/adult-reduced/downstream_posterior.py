import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import max_ent_inference as mei
from lib import bayesian_logistic_regression as blr
from lib.markov_network import MarkovNetworkJax, MarkovNetworkTorch
from lib.DataFrameData import DataFrameData
from lib import marginal_query
import adult_reduced_problem

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import arviz as az
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) 
torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "adult downstream posterior")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    with open(str(snakemake.input.napsu_mq_posterior), "rb") as file:
        posterior_obj = pickle.load(file)
    with open(str(snakemake.input.real_data_results), "rb") as file:
        real_data_obj = pickle.load(file)
    with open(str(snakemake.input.queries), "rb") as file:
        query_sets = pickle.load(file)

    marginalised_posterior_values = posterior_obj["marginalised_posterior_values"]
    reordered_columns = real_data_obj["reordered_columns"]

    cat_df = pd.read_csv(str(snakemake.input.data), dtype="category")
    df_data = DataFrameData(cat_df)
    queries = marginal_query.FullMarginalQuerySet(query_sets, df_data.values_by_col)
    queries = queries.get_canonical_queries()
    mn = MarkovNetworkTorch(df_data.values_by_col, queries)

    n_syn_datasets = 100
    n_syn_dataset = 10 * df_data.n
    # n_syn_datasets = 10
    # n_syn_dataset = 1 * df_data.n

    inds = np.random.choice(marginalised_posterior_values.shape[0], n_syn_datasets)
    posterior_sample = marginalised_posterior_values[inds, :]
    posterior_sample_torch = torch.tensor(np.array(posterior_sample))
    syn_int_dfs = [mn.sample(posterior_sample_torch[i], n_syn_dataset) for i in range(n_syn_datasets)]
    syn_dfs = [df_data.int_df_to_cat_df(syn_data) for syn_data in syn_int_dfs]

    transformed_syn_array = [sm.add_constant(adult_reduced_problem.df_transform(syn_df, reordered_columns)) for syn_df in syn_dfs]
    transformed_syn_array = [df.values for df in transformed_syn_array]

    # Doing these sequentially as the vmap takes too much memory
    syn_laplace_approxes = [
        blr.laplace_approximation_numpyro(rng, syn_vals) for syn_vals in transformed_syn_array#[0:1]
    ]

    result_obj = {
        "marginalised_laplace_approxes": syn_laplace_approxes,
        "epsilon": posterior_obj["epsilon"],
        "delta": posterior_obj["delta"],
        "repeat": posterior_obj["repeat"],
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(result_obj, file)