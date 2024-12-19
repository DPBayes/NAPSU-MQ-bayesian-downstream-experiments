import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import bayesian_logistic_regression as blr
import adult_reduced_problem

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) 
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat)
    seed = rng_initialisation.get_seed(repeat_ind, "adult downstream posterior synthpop")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    with open(str(snakemake.input.real_data_results), "rb") as file:
        real_data_obj = pickle.load(file)
    reordered_columns = real_data_obj["reordered_columns"]

    syn_dfs = []
    for syn_data_filename in snakemake.input.synthpop_syn_data:
        syn_df = pd.read_csv(str(syn_data_filename), index_col=False, dtype="category")
        syn_df.rename(columns=lambda col_name: col_name.replace(".", "-"), inplace=True) # Fix column names from synthpop
        syn_dfs.append(syn_df)


    transformed_syn_array = [sm.add_constant(adult_reduced_problem.df_transform(syn_df, reordered_columns)) for syn_df in syn_dfs]
    transformed_syn_array = [df.values for df in transformed_syn_array]

    # Doing these sequentially as the vmap takes too much memory
    syn_laplace_approxes = [
        blr.laplace_approximation_numpyro(rng, syn_vals) for syn_vals in transformed_syn_array#[0:1]
    ]

    result_obj = {
        "marginalised_laplace_approxes": syn_laplace_approxes,
        "repeat": repeat_ind,
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(result_obj, file)