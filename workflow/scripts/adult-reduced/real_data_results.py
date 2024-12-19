import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import binary_lr_data
from lib import bayesian_logistic_regression as blr
from lib import hierarchical_bayesian_logistic_regression as hblr
import adult_reduced_problem

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
import arviz as az
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) 
torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    seed = rng_initialisation.get_seed("real data adult reduced")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    cols_in_lr = ["income", "age", "race", "gender"]
    c_df = pd.read_csv(str(snakemake.input.discretised_data), dtype="category")[cols_in_lr]
    c_df["age_continuous"] = c_df.age.apply(lambda age: adult_reduced_problem.interval_middle(age)).astype(float)
    c_df.drop(columns=["age"], inplace=True)
    c_df["income"] = (c_df["income"] == "True").astype(int)
    oh_df = pd.get_dummies(c_df, dtype=int)
    oh_df.drop(columns=["race_White", "gender_Female"], inplace=True)
    reordered_columns = list(oh_df.columns[1:]) + [oh_df.columns[0]]
    reordered_oh_df = sm.add_constant(oh_df[reordered_columns])

    rng, key = jax.random.split(rng)
    nondp_laplace_approx = blr.laplace_approximation_numpyro(key, jnp.array(reordered_oh_df.values))

    reordered_oh_df = reordered_oh_df[["gender_Male", "const", "age_continuous", "income"]]
    data = jnp.array(reordered_oh_df.values)

    rng, key = jax.random.split(rng)
    mcmc = hblr.run_bayesian_logistic_regression_numpyro(
        key, data, n_levels=2, level_names={0: "F", 1: "M"},
        num_samples=1000, warmup_steps=500, num_chains=2, 
    )
    inf_data = az.from_numpyro(mcmc, log_likelihood=False)

    obj = {
        "n_orig": reordered_oh_df.shape[0],
        "reordered_columns": reordered_columns,
        "nondp_laplace_approx": nondp_laplace_approx,
        "nondp_hierarchical_inf_data": inf_data,
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(obj, file)