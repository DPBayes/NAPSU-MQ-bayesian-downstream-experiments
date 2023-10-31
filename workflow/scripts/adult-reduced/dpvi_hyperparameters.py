import sys
import os
sys.path.append(os.getcwd())
from lib import rng_initialisation
import adult_reduced_problem

import pandas as pd
import numpy as np
import pickle 
import statsmodels.api as sm

from lib import bayesian_logistic_regression as blr
from lib import optax_dpvi

from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp

import optuna


if __name__ == "__main__":

    epsilon = float(snakemake.wildcards.epsilon)
    seed = rng_initialisation.get_seed(epsilon, "adult dpvi")
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)

    cols_in_lr = ["income", "age", "race", "gender"]
    c_df = pd.read_csv(str(snakemake.input.discretised_data), dtype="category")[cols_in_lr]
    c_df["age_continuous"] = c_df.age.apply(lambda age: adult_reduced_problem.interval_middle(age)).astype(float) / 100
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

    rng_key, nondp_rng = jax.random.split(rng_key, 2)
    nondp_laplace_mean, nondp_laplace_cov = blr.laplace_approximation_numpyro(nondp_rng, jnp.array(reordered_oh_df.values))

    n_trials = 100
    # n_trials = 2
    tuning_keys = jax.random.split(rng_key, n_trials)

    def objective(trial):
        rng_key = tuning_keys[trial.number]
        # if (trial.number + 1) % 10 == 0:
        #     jax.clear_caches() # Fix memory leak, requires recent jax version

        q = trial.suggest_float("q", 0.001, 1.0)
        num_iter = trial.suggest_int("num_iter", int(1e4), int(1e5), log=True)
        # num_iter = trial.suggest_int("num_iter", int(1e2), int(5e2), log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        clipping_threshold = trial.suggest_float("clipping_threshold", 0.1, 50)

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
        mean = learned_parameters["m"]
        cov = jnp.diag(jnp.exp(learned_parameters["s"]))
        mean_error, kl_div = optax_dpvi.errors(mean, cov, nondp_laplace_mean, nondp_laplace_cov)
        return mean_error
        # return kl_div

    optuna_sampler = optuna.integration.BoTorchSampler()
    study_name = "dpvi-hyperparameters-{}".format(epsilon)
    storage_name = "sqlite:///results/adult-reduced/checkpoints/{}.db".format(study_name)
    study = optuna.create_study(
        sampler=optuna_sampler, study_name=study_name, storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials - len(study.trials))

    result_obj = {
        "study": study,
        "epsilon": epsilon,
        "delta": delta,
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(result_obj, file)