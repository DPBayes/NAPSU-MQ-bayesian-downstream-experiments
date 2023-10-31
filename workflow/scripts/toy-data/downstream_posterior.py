import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import max_ent_inference as mei
from lib import bayesian_logistic_regression as blr
import lib.max_ent_dist as med

import functools
import numpy as np
import arviz as az
import torch
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) 
config.update('jax_platform_name', 'cpu')
torch.set_default_dtype(torch.float64)


def generate_syn_datasets_lambdas(posterior_values, n_syn_dataset, n_syn_datasets, progressbar=True):
    inds = np.random.choice(posterior_values.shape[0], n_syn_datasets)
    posterior_sample = posterior_values[inds, :]
    return mei.generate_synthetic_data(posterior_sample, n_syn_dataset, max_ent_dist, show_progressbar=progressbar)

if __name__ == "__main__":
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "downstream posterior")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    with open(str(snakemake.input.napsu_mq_posterior), "rb") as file:
        posterior_obj = pickle.load(file)
    with open(str(snakemake.input.real_data_results), "rb") as file:
        real_data_obj = pickle.load(file)

    x_values = real_data_obj["x_values"]
    values_by_feature = real_data_obj["values_by_feature"]
    n_orig = real_data_obj["n_orig"]
    marginalised_posterior_values = posterior_obj["marginalised_posterior_values"]
    unmarginalised_posterior_s = posterior_obj["unmarginalised_posterior_s"]
    queries = posterior_obj["queries"]
    max_ent_dist = med.MaximumEntropyDistribution(values_by_feature, queries)

    n_syn_datasets_vals = [25, 50, 100, 200, 400]
    n_syn_dataset_mul_vals = [1, 2, 5, 10, 20]

    all_laplace_approxes = {}
    for n_syn_datasets_val in n_syn_datasets_vals:
        for n_syn_dataset_mul in n_syn_dataset_mul_vals:
            n_syn_dataset_val = n_orig * n_syn_dataset_mul
            syn_datasets = generate_syn_datasets_lambdas(marginalised_posterior_values, n_syn_dataset_val, n_syn_datasets_val, False)
            rng, key = jax.random.split(rng, 2)
            laplace_approxes = blr.syn_data_bayes_lr_laplace_approx(key, jnp.array(syn_datasets))
            all_laplace_approxes[(n_syn_datasets_val, n_syn_dataset_mul)] = laplace_approxes


    jax_x_values = jnp.array(x_values)
    all_query_values = jnp.concatenate((jnp.array(queries(x_values)), jnp.array((1, 0, 0, 0, 0, 0, 0, 0)).reshape((-1, 1))), axis=1)
    x_to_s_indexes = all_query_values.argmax(axis=1)

    def dataset_from_suff_stat(suff_stat):
        one_element_syn_datasets = [
            jnp.tile(jax_x_values[i, :], (suff_stat[x_to_s_indexes[i]], 1)) 
            for i in range(jax_x_values.shape[0])
        ]
        dataset = jnp.concatenate(one_element_syn_datasets, axis=0)
        return dataset

    n_syn_datasets_unmarginalised = 100
    inds = np.random.choice(unmarginalised_posterior_s.shape[0], n_syn_datasets_unmarginalised)
    posterior_s_sample = unmarginalised_posterior_s[inds, :]
    s_posterior_syn_datasets = jnp.stack([
        dataset_from_suff_stat(s.astype(int))
        for s in posterior_s_sample
    ])
    s_posterior_laplace_approxes = blr.syn_data_bayes_lr_laplace_approx(rng, jnp.array(s_posterior_syn_datasets))

    result_obj = {
        "s_posteriors": s_posterior_laplace_approxes,
        "marginalised_laplace_approxes": all_laplace_approxes,
        "epsilon": posterior_obj["epsilon"],
        "delta": posterior_obj["delta"],
        "repeat": posterior_obj["repeat"],
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(result_obj, file)