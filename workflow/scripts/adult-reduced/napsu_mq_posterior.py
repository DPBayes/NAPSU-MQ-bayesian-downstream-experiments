import sys
import os
sys.path.append(os.getcwd())
import pickle
import time
from lib import rng_initialisation
from lib import max_ent_inference as mei
from lib import max_ent_inference_unmarginalised as meiu
from lib import marginal_query
from lib.markov_network import MarkovNetworkJax, MarkovNetworkTorch
from lib.DataFrameData import DataFrameData
import lib.max_ent_dist as med
from lib import privacy_accounting

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
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "adult napsu-mq")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    cat_df = pd.read_csv(str(snakemake.input.data), dtype="category")
    df_data = DataFrameData(cat_df)
    int_tensor = df_data.int_tensor

    with open(str(snakemake.input.queries), "rb") as file:
        query_sets = pickle.load(file)

    queries = marginal_query.FullMarginalQuerySet(query_sets, df_data.values_by_col)
    queries = queries.get_canonical_queries()

    epsilon = float(snakemake.wildcards.epsilon)
    delta = df_data.n ** (-2)
    sensitivity = np.sqrt(2 * len(query_sets))
    sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

    mntorch = MarkovNetworkTorch(df_data.values_by_col, queries)
    mnjax = MarkovNetworkJax(df_data.values_by_col, queries)

    suff_stat = torch.sum(queries.flatten()(int_tensor), axis=0)
    dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP)

    lap_start_time = time.time()
    laplace_approx, lap_losses, fail_count = mei.laplace_approximation_normal_prior(dp_suff_stat, df_data.n, sigma_DP, mntorch, max_retries=5, max_iters=6000)
    lap_end_time = time.time()
    rng, mcmc_key = jax.random.split(rng)
    mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
        mcmc_key, dp_suff_stat, df_data.n, sigma_DP, mnjax, laplace_approx, num_samples=2000, num_warmup=800, num_chains=4, disable_progressbar=True
        # mcmc_key, dp_suff_stat, df_data.n, sigma_DP, mnjax, laplace_approx, num_samples=2, num_warmup=8, num_chains=2, disable_progressbar=True
    )
    inf_data = az.from_numpyro(mcmc, log_likelihood=False)
    mcmc_end_time = time.time()
    mcmc_table = az.summary(inf_data)
    posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
    posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())

    diagnostics = {"losses": lap_losses, "fails": fail_count, "mcmc_table": mcmc_table}
    runtime = {"laplace": lap_end_time - lap_start_time, "mcmc": mcmc_end_time - lap_end_time}

    save_obj = {
        "marginalised_posterior_values": posterior_values,
        "n_orig": df_data.n,
        "epsilon": epsilon,
        "delta": delta,
        "repeat": int(snakemake.wildcards.repeat),
        "diagnostics": diagnostics,
        "runtime": runtime,
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(save_obj, file)