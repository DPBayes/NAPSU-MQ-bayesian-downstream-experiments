import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import binary_lr_data
from lib import bayesian_logistic_regression as blr

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) 
config.update('jax_platform_name', 'cpu')
torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "real data")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    true_params = torch.tensor((1.0, 0.0))
    data_gen = binary_lr_data.BinaryLogisticRegressionDataGenerator(true_params)
    data = data_gen.generate_data(2000)
    x_values = data_gen.x_values
    n, d = data.shape

    rng, key = jax.random.split(rng)
    nondp_post = blr.laplace_approximation_numpyro(rng, jnp.array(data))

    obj = {
        "data": data,
        "x_values": x_values,
        "values_by_feature": data_gen.values_by_feature,
        "n_orig": n,
        "true_params": true_params,
        "nondp_post": nondp_post,
        "repeat": int(snakemake.wildcards.repeat),
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(obj, file)