import sys
import os
sys.path.append(os.getcwd())
import pickle
import itertools
from lib import rng_initialisation
from lib import max_ent_inference as mei
from lib import hierarchical_bayesian_logistic_regression as hblr
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
    syn_data_ind = int(snakemake.wildcards.syn_data_ind)
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), float(snakemake.wildcards.epsilon), syn_data_ind, "adult hierarchical downstream posterior")
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
    columns_in_queries = set(itertools.chain(*query_sets))
    columns_not_in_queries = [col for col in cat_df.columns if col not in columns_in_queries]
    cat_df.drop(columns=columns_not_in_queries, inplace=True)

    df_data = DataFrameData(cat_df)
    queries = marginal_query.FullMarginalQuerySet(query_sets, df_data.values_by_col)
    queries = queries.get_canonical_queries()
    mn = MarkovNetworkTorch(df_data.values_by_col, queries)

    n_syn_dataset = 10 * df_data.n
    # n_syn_dataset = 1 * df_data.n

    inds = np.random.choice(marginalised_posterior_values.shape[0], 1)
    posterior_sample = marginalised_posterior_values[inds, :]
    posterior_sample_torch = torch.tensor(np.array(posterior_sample))
    syn_int_df = mn.sample(posterior_sample_torch[0], n_syn_dataset)
    syn_df = df_data.int_df_to_cat_df(syn_int_df)


    transformed_syn_array = sm.add_constant(adult_reduced_problem.df_transform(syn_df, reordered_columns))
    transformed_syn_array = transformed_syn_array[["gender_Male", "const", "age_continuous", "income"]]
    transformed_syn_array = jnp.array(transformed_syn_array.values)

    rng, key = jax.random.split(rng)
    mcmc = hblr.run_bayesian_logistic_regression_numpyro(
        key, transformed_syn_array, n_levels=2, level_names={0: "F", 1: "M"},
        num_samples=1000, warmup_steps=500, num_chains=2, 
        # num_samples=15, warmup_steps=5, num_chains=2, 
    )
    inf_data = az.from_numpyro(mcmc, log_likelihood=False)
    
    result_obj = {
        "hierarchical_inf_data": inf_data,
        "epsilon": posterior_obj["epsilon"],
        "delta": posterior_obj["delta"],
        "repeat": posterior_obj["repeat"],
        "syn_data_ind": syn_data_ind,
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(result_obj, file)