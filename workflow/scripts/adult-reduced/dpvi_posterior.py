import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
import adult_reduced_problem

import pandas as pd
import numpy as np
from scipy import stats
import torch
import statsmodels.api as sm

from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp

from lib import optax_dpvi


if __name__ == "__main__":
    epsilon = float(snakemake.wildcards.epsilon)
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), epsilon, "adult dpvi")
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)

    dpvi_age_divisor = 100
    age_coef_index = 1

    cols_in_lr = ["income", "age", "race", "gender"]
    c_df = pd.read_csv(str(snakemake.input.discretised_data), dtype="category")[cols_in_lr]
    c_df["age_continuous"] = c_df.age.apply(lambda age: adult_reduced_problem.interval_middle(age)).astype(float) / dpvi_age_divisor
    c_df.drop(columns=["age"], inplace=True)
    c_df["income"] = (c_df["income"] == "True").astype(int)
    oh_df = pd.get_dummies(c_df, dtype=int)
    oh_df.drop(columns=["race_White", "gender_Female"], inplace=True)
    reordered_columns = list(oh_df.columns[1:]) + [oh_df.columns[0]]
    reordered_oh_df = sm.add_constant(oh_df[reordered_columns])
    n, d = c_df.shape
    data = reordered_oh_df.drop(columns=["income"]).values
    labels = reordered_oh_df["income"].values
    X = jnp.array(data)
    y = jnp.array(labels)

    delta = n**(-2)

    with open(str(snakemake.input.hyperparameters), "rb") as file:
        hyper_obj = pickle.load(file)
    
    best_params = hyper_obj["study"].best_params
    # best_params = {"q": 0.05, "num_iter": 100000, "lr": 1e-3, "clipping_threshold": 2.0}

    q = best_params["q"]
    num_iter = best_params["num_iter"]
    lr = best_params["lr"]
    clipping_threshold = best_params["clipping_threshold"]

    dpvi_args = optax_dpvi.DPVIArgs(
        seed=rng_key,
        epsilon=epsilon,
        delta=delta,
        num_iterations=num_iter,
        sampling_ratio=q,
        clipping_threshold=clipping_threshold,
        learning_rate=lr,
    )

    learned_parameters, opt_state = optax_dpvi.dpvi_train(X, y, 1, dpvi_args)

    dpvi_mean = learned_parameters["m"]
    dpvi_mean = dpvi_mean.at[age_coef_index].divide(dpvi_age_divisor)

    dpvi_cov = jnp.diag(jnp.exp(learned_parameters["s"]))
    dpvi_cov = dpvi_cov.at[age_coef_index, age_coef_index].divide(dpvi_age_divisor**2)

    save_obj = {
        "dpvi_posterior": (dpvi_mean, dpvi_cov),
        "n_orig": n,
        "epsilon": epsilon,
        "delta": delta,
        "repeat": int(snakemake.wildcards.repeat),
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(save_obj, file)