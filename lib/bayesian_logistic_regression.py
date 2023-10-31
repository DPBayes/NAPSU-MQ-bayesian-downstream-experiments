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

def bayesian_logistic_regression(data):
    n, d = data.shape
    X = data[:, 0:d-1]
    y = data[:, d-1]
    xdim = d - 1

    theta = pyro.sample("theta", dist.MultivariateNormal(torch.zeros(xdim), 10 * torch.eye(xdim)))
    alpha = torch.mv(X, theta)
    return pyro.sample("y", dist.Bernoulli(logits=alpha), obs=y)

def run_bayesian_logistic_regression(
    data, num_samples=1000, warmup_steps=300, num_chains=1, 
    disable_progressbar=False
    ):
    kernel = pyro.infer.NUTS(bayesian_logistic_regression)
    mcmc = pyro.infer.MCMC(
        kernel, num_samples=num_samples, num_chains=num_chains, 
        warmup_steps=warmup_steps, disable_progbar=disable_progressbar, 
        mp_context="forkserver"
    )
    mcmc.run(data)
    return mcmc

def bayesian_logistic_regression_numpyro(data, tempering=1):
    n, d = data.shape
    X = data[:, 0:d-1]
    y = data[:, d-1]
    xdim = d - 1

    theta = numpyro.sample("theta", npdist.MultivariateNormal(jnp.zeros(xdim), 10 * jnp.eye(xdim)))
    alpha = jnp.dot(X, theta)
    return numpyro.sample("y", npdist.Binomial(total_count=tempering, logits=alpha), obs=y * tempering)

def run_bayesian_logistic_regression_numpyro(
    rng, data, num_samples=1000, warmup_steps=300, num_chains=1, 
    disable_progressbar=False, tempering=1
    ):
    kernel = numpyro.infer.NUTS(bayesian_logistic_regression_numpyro)
    mcmc = numpyro.infer.MCMC(
        kernel, num_samples=num_samples, num_chains=num_chains, 
        num_warmup=warmup_steps, progress_bar=not disable_progressbar, 
    )
    mcmc.run(rng, jnp.array(data), tempering)
    return mcmc

def laplace_approximation_numpyro(rng, data, tempering=1):
    init_param, potential_fn_, _, __ = numpyro.infer.util.initialize_model(
        rng, bayesian_logistic_regression_numpyro, dynamic_args=True,
        model_args=(jnp.array(data), tempering)
    )
    potential_fn = jax.jit(lambda theta, data: potential_fn_(data, tempering)(theta))
    return laplace_approximation_from_potential(potential_fn, data, init_param.z)

def laplace_approximation_from_potential(potential_fn, data, init_param):
    solver = jaxopt.LBFGS(fun=potential_fn, maxiter=1000)
    solution, diagnostics = solver.run(init_param, data)
    hess = jax.hessian(potential_fn)(solution, data)["theta"]["theta"]
    cov = jnp.linalg.inv(hess)
    mean = solution["theta"]
    return mean, cov

def syn_data_bayes_lr(rng, syn_datasets, num_samples, num_chains):
    n_syn_datasets, n_syn_dataset, d = syn_datasets.shape
    syn_posteriors = np.zeros((n_syn_datasets, num_samples * num_chains, d - 1))
    r_hats = np.zeros((n_syn_datasets, d - 1))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="There are not enough devices to run parallel chains")
        for i in tqdm(range(n_syn_datasets)):
            rng, key = jax.random.split(rng)
            syn_dataset = jnp.array(syn_datasets[i, :, :])
            mcmc = run_bayesian_logistic_regression_numpyro(
                rng, syn_dataset, num_samples=num_samples, num_chains=num_chains, 
                disable_progressbar=True
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            syn_posteriors[i, :, :] = inf_data.posterior.stack(draws=("draw", "chain")).theta.values.transpose()
            r_hats[i, :] = az.rhat(inf_data).theta.values
    
    return syn_posteriors, r_hats

def syn_data_bayes_lr_laplace_approx(rng, syn_datasets, tempering=1):
    init_param, potential_fn_, _, __ = numpyro.infer.util.initialize_model(
        rng, bayesian_logistic_regression_numpyro, dynamic_args=True,
        model_args=(syn_datasets[0, :, :], tempering)
    )
    potential_fn = lambda theta, data: potential_fn_(data, tempering)(theta)

    partial_pot_fn = jax.jit(functools.partial(laplace_approximation_from_potential, potential_fn))
    vmap_partial_pot_fn = jax.vmap(partial_pot_fn, (0, None), 0)
    means, covs = vmap_partial_pot_fn(syn_datasets, init_param.z)
    return means, covs
