import torch
import pyro
import pyro.distributions as dist
import jax
import jax.numpy as jnp 
import numpyro 
import numpyro.distributions as npdist
import numpy as np
from scipy import stats
import jaxopt
import warnings
import functools
from tqdm import tqdm
import arviz as az


def hierarchical_bayesian_logistic_regression_numpyro(data, n_levels=2, level_names=None):
    if level_names is None:
        level_names = {i: i for i in range(n_levels)}
    n, d = data.shape
    levels = data[:, 0].astype(int)
    X = data[:, 1:d-1]
    y = data[:, d-1]
    xdim = d - 2

    mu = numpyro.sample("mu", npdist.MultivariateNormal(jnp.zeros(xdim), 10 * jnp.eye(xdim)))
    v = numpyro.sample("v", npdist.MultivariateNormal(jnp.zeros(xdim), jnp.eye(xdim)))
    tau = numpyro.deterministic("tau", jnp.exp(v))
    thetas = [
        numpyro.sample("gender: {}".format(level_names[i]), npdist.MultivariateNormal(mu, jnp.diag(tau**2)))
        for i in range(n_levels)
    ]
    theta = jnp.stack(thetas, axis=0)
    selected_theta = theta[levels, :]
    alpha = (X * selected_theta).sum(axis=1)
    return numpyro.sample("y", npdist.Binomial(total_count=1, logits=alpha), obs=y)

def run_bayesian_logistic_regression_numpyro(
    rng, data, n_levels=2, level_names=None, num_samples=1000, warmup_steps=300, num_chains=1, 
    disable_progressbar=False,
    ):
    kernel = numpyro.infer.NUTS(hierarchical_bayesian_logistic_regression_numpyro)
    mcmc = numpyro.infer.MCMC(
        kernel, num_samples=num_samples, num_chains=num_chains, 
        num_warmup=warmup_steps, progress_bar=not disable_progressbar, 
    )
    mcmc.run(rng, jnp.array(data), n_levels, level_names)
    return mcmc

def funnel_dist(dim=2):
    v = numpyro.sample("v", npdist.Normal(0.0, 1.0))
    tau = numpyro.deterministic("tau", jnp.exp(v))
    thetas = [
        numpyro.sample("theta{}".format(i), npdist.Normal(0, tau))
        for i in range(dim)
    ]
    return thetas

def run_funnel_dist(
    rng, num_samples=1000, warmup_steps=300, num_chains=1, 
    disable_progressbar=False,
    ):
    kernel = numpyro.infer.NUTS(funnel_dist)
    mcmc = numpyro.infer.MCMC(
        kernel, num_samples=num_samples, num_chains=num_chains, 
        num_warmup=warmup_steps, progress_bar=not disable_progressbar, 
    )
    mcmc.run(rng)
    return mcmc