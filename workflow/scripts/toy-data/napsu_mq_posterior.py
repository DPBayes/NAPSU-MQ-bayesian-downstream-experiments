import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import max_ent_inference as mei
from lib import max_ent_inference_unmarginalised as meiu
from lib import marginal_query
import lib.max_ent_dist as med
from lib import privacy_accounting

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


if __name__ == "__main__":
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "napsu-mq")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    with open(str(snakemake.input), "rb") as file:
        obj = pickle.load(file)

    data = obj["data"]
    values_by_feature = obj["values_by_feature"]
    n, d = data.shape

    fmqs = marginal_query.FullMarginalQuerySet([(0, 1, 2)], values_by_feature)
    canonical_queries = fmqs.get_canonical_queries()
    queries = canonical_queries.flatten()
    max_ent_dist = med.MaximumEntropyDistribution(values_by_feature, queries)

    suff_stat = torch.sum(queries(data), axis=0)
    sensitivity = np.sqrt(2)
    epsilon = float(snakemake.wildcards.epsilon)
    delta = n**(-2)
    sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
    dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP)


    num_samples = 1500 if epsilon == 0.1 else 500
    mcmc, backtransform = mei.run_mcmc_normalised(
        dp_suff_stat, n, sigma_DP, max_ent_dist, 
        num_samples=num_samples, num_chains=4, disable_progressbar=True
    )
    inf_data = az.from_pyro(mcmc, log_likelihood=False)
    marginalised_posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
    marginalised_posterior_values = backtransform(marginalised_posterior_values.norm_lambdas.values.transpose())


    lambdas_proposal_step_size = 0.05
    lambdas_proposal_num_steps = 20
    s_proposal_iters = 30
    num_iters = 20000
    num_chains = 4

    rng, init_key, mcmc_key = jax.random.split(rng, 3)
    init_value = meiu.get_init_values(init_key, dp_suff_stat, 50, n, num_chains)

    log_target = functools.partial(
        meiu.unmarginalised_log_target, dp_suff_stat=jnp.array(dp_suff_stat), 
        n=n, sigma_DP=sigma_DP
    )
    full_samples, accepts = meiu.metropolis_hastings(
        mcmc_key, log_target, init_value, lambdas_proposal_step_size, 
        lambdas_proposal_num_steps, s_proposal_iters,
        num_iters, num_chains, display_progressbar=False
    )
    samples = jax.tree_util.tree_map(lambda samp: samp[:, int(num_iters * 0.2):, :], full_samples)
    inf_data = az.from_dict(samples)
    unmarginalised_posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
    unmarginalised_posterior_lambdas = unmarginalised_posterior_values.lambdas.values.transpose()
    unmarginalised_posterior_s = unmarginalised_posterior_values.s.values.transpose()

    save_obj = {
        "marginalised_posterior_values": marginalised_posterior_values,
        "unmarginalised_posterior_s": unmarginalised_posterior_s,
        "unmarginalised_posterior_lambdas": unmarginalised_posterior_lambdas,
        "n_orig": n,
        "queries": queries,
        "epsilon": epsilon,
        "delta": delta,
        "repeat": obj["repeat"],
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(save_obj, file)