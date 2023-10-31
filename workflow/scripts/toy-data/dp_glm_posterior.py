import sys
import os
sys.path.append(os.getcwd())
import pickle
from lib import rng_initialisation
from lib import dp_glm_interface as dpglm

import numpy as np
import arviz as az


if __name__ == "__main__":
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat), "dp-glm")
    np.random.seed(seed)

    with open(str(snakemake.input), "rb") as file:
        obj = pickle.load(file)

    data = obj["data"]
    n, d = data.shape

    epsilon = float(snakemake.wildcards.epsilon)
    delta = n**(-2)

    N_mcmc_samples= 1000
    R=(d-1)**0.5     ### || x ||_2 <= R.
    s = 5
    X_train = data[:, 0:d-1].numpy()
    y_train = data[:, d-1].numpy()
    with dpglm.suppress_stdout_stderr():
        fit = dpglm.run_dp_glm(X_train, y_train, epsilon, delta, R, s, N_mcmc_samples)

    save_obj = {
        "dp_glm_inf_data": az.from_pystan(fit),
        "n_orig": n,
        "epsilon": epsilon,
        "delta": delta,
        "repeat": obj["repeat"],
    }
    with open(str(snakemake.output), "wb") as file:
        pickle.dump(save_obj, file)