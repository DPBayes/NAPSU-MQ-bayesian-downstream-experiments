import jax, optax, tqdm, argparse
 
import numpy as np
import jax.numpy as jnp
import pandas as pd
import statsmodels.api as sm
 
from optax._src.combine import chain
from optax._src.transform import scale_by_adam, scale
from optax import differentially_private_aggregate
from numpyro.distributions import Bernoulli, Normal
from d3p.dputil import approximate_sigma
from collections import namedtuple
 
DPVIArgs = namedtuple(
    "DPVIArgs",
    [
        "seed",
        "epsilon",
        "delta",
        "num_iterations",
        "sampling_ratio",
        "clipping_threshold",
        "learning_rate",
    ],
)
 
#######
# Load data
def load_data(scale_age=False):
    dtypes = {
        "age": int,
        "workclass": "category",
        "education": "category",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "gender": "category",
        "capital-gain": int,
        "capital-loss": int,
        "hours-per-week": int,
        "native-country": "category",
        "income": bool,
    }
    df = pd.read_csv(
        "datasets/adult.csv",
        na_values=["?"],
        dtype=dtypes,
        true_values=[">50K"],
        false_values=["<=50K"],
    )
 
    # preprocess
    df.drop(
        columns=[
            "fnlwgt",
            "educational-num",
            "native-country",
            "occupation",
            "relationship",
        ],
        inplace=True,
    )
    df.dropna(inplace=True)
    df["age"] = pd.cut(df["age"], 5)
    df["hours-per-week"] = pd.cut(df["hours-per-week"], 5)
    df["capital-gain"] = (df["capital-gain"] > 0).astype("category")
    df["capital-loss"] = (df["capital-loss"] > 0).astype("category")
 
    # choose columns for logistic regression
    cols_in_lr = ["income", "age", "race", "gender"]
    c_df = df.copy()[cols_in_lr]
    c_df["age_continuous"] = c_df.age.apply(lambda age: age.mid).astype(float)
    if scale_age:
        c_df["age_continuous"] = c_df.age_continuous.apply(lambda age: age / 100.).astype(float)
    c_df.drop(columns=["age"], inplace=True)
    c_df["income"] = c_df["income"].astype(int)
    oh_df = pd.get_dummies(c_df, dtype=int)
    oh_df.drop(columns=["race_White", "gender_Female"], inplace=True)
 
    #
    reordered_columns = list(oh_df.columns[1:]) + [oh_df.columns[0]]
    reordered_oh_df = sm.add_constant(oh_df[reordered_columns])
 
    return reordered_oh_df
 
 
#######
# define elbo for logistic regression
 
def log_likelihood_loss(w, X, y):
    logits = X @ w
    log_probs = Bernoulli(logits=logits).log_prob(y)
    return -1.0 * log_probs.sum()
 
 
def log_prior_loss(w):
    log_probs = Normal(0.0, 10.0**0.5).log_prob(w)
    return -1.0 * log_probs.sum()
 
 
def entropy_loss(s):
    """
    assuming the exp transform
    """
    d = s.shape[0]
    return -1.0 * (0.5 * d * jnp.log(2 * jnp.pi * jnp.exp(1.0)) + s.sum())
 
 
def elbo_loss(params, rng, X, y, scaling=1.0):
    """
    NOTE: When running DP-SGD with optax, we vmap this function
    over the batch_size (B) samples. Hence, summing these gradients
    would have both entropy and log-prior terms B times. We can
    avoid this by scaling the terms by scaling = batch_size.
    """
    w = params["m"] + jnp.exp(params["s"]) * jax.random.normal(
        rng, shape=params["m"].shape
    )
    return (
        log_likelihood_loss(w, X, y)
        + log_prior_loss(w) / scaling
        + entropy_loss(params["s"]) / scaling
    )
 
def dpvi_train(X, y, num_chunks, args):
    """
    num_chunks (int): defines how often we want print to observe progress.
                      Setting this to 1 is most likely the fastest in terms
                      of JAX
    args (NamedTuple): a namedtuple containing parameters for DPVI
    """
 
    rng = args.seed
    epsilon = args.epsilon
    delta = args.delta
    clipping_threshold = args.clipping_threshold
    q = args.sampling_ratio
    num_iter = args.num_iterations
    learning_rate = args.learning_rate
 
    N, d = X.shape
    batch_size = int(q * N)
    ## find the noise multiplier
    noise_multiplier, _, _ = approximate_sigma(epsilon, delta, q, num_iter)
 
    # rng = jax.random.PRNGKey(seed)
    rng, dp_rng = jax.random.split(rng)
    ## set the optimizer
    dp_sgd_agg = differentially_private_aggregate(
        # l2_norm_clip=clipping_threshold, noise_multiplier=noise_multiplier, seed=jax.random.bits(dp_rng, dtype=jnp.uint32)
        l2_norm_clip=clipping_threshold, noise_multiplier=noise_multiplier, seed=jax.random.randint(dp_rng, (1,), 0, 2**32, dtype=jnp.uint32).item()
    )
    optimizer = chain(
        dp_sgd_agg,
        scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None),
        scale(-learning_rate),
    )
    rng, init_rng = jax.random.split(rng)
    step_rngs = jax.random.split(rng, num_iter)
 
    ## initialize variational parameters and optimizer
    init_params = {
        "m": 0.1 * jax.random.normal(init_rng, shape=(d,)),
        "s": jnp.log(0.01) * jnp.ones(d),
    }
 
    params = init_params.copy()
    opt_state = optimizer.init(params)
 
    ## define a function to do a single DPVI step
    def take_step(t, val):
        opt_state, params = val
 
        # batch_rng, vi_rng = jax.random.split(jax.random.PRNGKey(seed + t), 2)
        batch_rng, vi_rng = jax.random.split(step_rngs[t], 2)
 
        ### sample minibatch
        batch_indices = jax.random.choice(
            batch_rng, N, shape=(batch_size,), replace=False,
        )
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
 
        ### compute per-example grads
        loss, grads = jax.vmap(
            jax.value_and_grad(elbo_loss), in_axes=(None, None, 0, 0, None)
        )(params, vi_rng, batch_X, batch_y, batch_size)
 
        ### update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
 
        return opt_state, params
 
    ## iterate over num_chunks
    steps_in_chunk = num_iter // num_chunks
    # use_tqdm = num_chunks != 1
    # for t in (pbar := tqdm.tqdm(range(num_chunks))):
    for t in range(num_chunks):
        start = t * steps_in_chunk  # start indx for the fori_loop
        end = (t + 1) * steps_in_chunk  # end indx for the fori_loop
        opt_state, params = jax.lax.fori_loop(
            start, end, take_step, (opt_state, params)
        )
 
        ## print out the elbo after each chunk
        loss = elbo_loss(params, jax.random.PRNGKey(0), X, y)
        # if use_tqdm:
        # pbar.set_description(f"loss: {loss.mean()}")
 
    return params, opt_state

def gaussian_kl_div(mean1, cov1, mean2, cov2):
    cov2inv = jnp.linalg.inv(cov2)
    term1 = jnp.trace(cov2inv @ cov1)
    term2 = (mean2 - mean1).reshape((1, -1)) @ cov2inv @ (mean2 - mean1).reshape((-1, 1))
    term3 = jnp.log(jnp.linalg.det(cov2) / jnp.linalg.det(cov1))
    return 0.5 * (term1 + term2 + term3 - mean1.shape[0])

def errors(dpvi_mean, dpvi_cov, nondp_laplace_mean, nondp_laplace_cov):
    mean_error = ((dpvi_mean - nondp_laplace_mean)**2).sum().item()
    kl_div = gaussian_kl_div(nondp_laplace_mean, nondp_laplace_cov, dpvi_mean, dpvi_cov).item()
    return mean_error, kl_div

 
def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument("seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_chunks", type=int, default=10, help="How many chunks for DPVI")
    parser.add_argument(
        "--scale_age",
        action="store_true",
        default=False,
        help="Whether to scale the age feature",
    )
 
    args, _ = parser.parse_known_args()
    # load data, split regressors and targets
    reordered_oh_df = load_data(scale_age = args.scale_age)
 
    data = reordered_oh_df.drop(columns=["income"]).values
    labels = reordered_oh_df["income"].values
 
    X = jnp.array(data)
    y = jnp.array(labels)
 
 
    #######
    # train model using DPVI
 
    dpvi_args = DPVIArgs(
        seed=args.seed,
        epsilon=1.0,
        delta=1 / data.shape[0] ** 2,
        num_iterations=100000,
        sampling_ratio=0.05,
        clipping_threshold=2.0,
        learning_rate=1e-3,
    )
 
 
    learned_parameters, opt_state = dpvi_train(X, y, args.num_chunks, dpvi_args)
    print(learned_parameters)
 
if __name__ == "__main__":
    main()
