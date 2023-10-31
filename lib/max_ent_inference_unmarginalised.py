import jax
import jax.numpy as jnp
import numpyro 
import numpyro.distributions as dist
import functools
from tqdm import tqdm

def tqdm_choice(iterable, use_tqdm):
    if use_tqdm:
        return tqdm(iterable)
    else:
        return iterable

@functools.partial(jax.jit, static_argnames="iters")
def s_proposal(s_proposal_key, proposal, iters=1):
    dim = proposal["s"].shape[1]
    num_chains = proposal["s"].shape[0]
    keys = jax.random.split(s_proposal_key, iters)

    def inner_body(carry, j):
        proposal, inc_index, dec_index = carry
        proposal["s"] = proposal["s"].at[j, inc_index[j]].add(1)
        proposal["s"] = proposal["s"].at[j, dec_index[j]].add(-1)
        return (proposal, inc_index, dec_index), None

    def outer_body(i, proposal):
        inc_key, dec_key = jax.random.split(keys[i], 2)
        inc_index = jax.random.randint(inc_key, (num_chains,), 0, dim)
        dec_index = jax.random.randint(dec_key, (num_chains,), 0, dim)
        return jax.lax.scan(inner_body, (proposal, inc_index, dec_index), jnp.arange(num_chains))[0][0]

    return jax.lax.fori_loop(0, iters, outer_body, proposal)

@jax.jit
def batch_dot(a, b):
    return (a @ b.transpose()).diagonal()

def lambdas_hmc_proposal(lambdas_proposal_key, gradient, proposal, step_size, num_steps):
    num_chains, dim = proposal["lambdas"].shape
    momentum = jax.random.normal(lambdas_proposal_key, (num_chains, dim))
    kin_energy_start = batch_dot(momentum, momentum) / 2
    for i in range(num_steps):
        momentum = momentum + step_size * gradient(proposal) / 2
        proposal["lambdas"] = proposal["lambdas"] + momentum * step_size
        momentum = momentum + step_size * gradient(proposal) / 2

    kin_energy_end = batch_dot(momentum, momentum) / 2
    kin_energy_diff = kin_energy_start - kin_energy_end
    return proposal, kin_energy_diff

@jax.jit
def unmarginalised_log_target(params, dp_suff_stat, n, sigma_DP):
    d = dp_suff_stat.shape[0]
    lambdas = params["lambdas"]
    lambdas_transformed = jnp.concatenate((lambdas, jnp.zeros(1)))
    s = params["s"]
    log_prior = dist.MultivariateNormal(jnp.zeros(d), 10 * jnp.eye(d)).log_prob(lambdas)
    log_likelihood_s = dist.Multinomial(total_count=n, logits=lambdas_transformed).log_prob(s)
    log_likelihood_s_tilde = dist.MultivariateNormal(s[0:-1], sigma_DP**2 * jnp.eye(d)).log_prob(dp_suff_stat)
    return log_prior + log_likelihood_s + log_likelihood_s_tilde

def accept_mul_into_shape(accepts, shape):
    num_extra_dims = len(shape) - 1
    return jnp.expand_dims(accepts, tuple(range(1, num_extra_dims + 1)))

def accept_test(accept_key, log_likelihood_ratio_vmap, current, proposal, hastings_term=0):
    num_chains = current["lambdas"].shape[0]
    ll_ratio = log_likelihood_ratio_vmap(current, proposal)
    uniform_values = jax.random.uniform(accept_key, shape=(num_chains,))
    accepts = 1 * (ll_ratio + hastings_term > jnp.log(uniform_values))
    updates = jax.tree_util.tree_map(
        lambda current_val, proposal_val: 
        accept_mul_into_shape(accepts, proposal_val.shape) * proposal_val + (1 - accept_mul_into_shape(accepts, current_val.shape)) * current_val,
        current, proposal
    )
    return updates, accepts


def metropolis_hastings(
    rng, log_target, init_value, lambdas_step_size, lambdas_num_steps, 
    s_proposal_iters, num_iters, num_chains, display_progressbar=True
    ):
    dims = jax.tree_util.tree_map(lambda val: val.shape[1], init_value)

    log_likelihood_ratio_vmap = jax.jit(jax.vmap(
        lambda current, proposal: log_target(proposal) - log_target(current),
        (0, 0), 0
    ))
    gradient = jax.grad(lambda lambdas, s: log_target({"lambdas": lambdas, "s": s}))
    gradient_vmap = jax.jit(jax.vmap(
        lambda params: gradient(params["lambdas"], params["s"]),
        0, 0
    ))

    acceptances = jax.tree_util.tree_map(lambda val: jnp.zeros((num_chains, num_iters)), init_value)
    samples = jax.tree_util.tree_map(
        lambda val: jnp.zeros((num_chains, num_iters + 1, val.shape[1])).at[:, 0, :].set(val), 
        init_value
    )

    rng, *proposal_keys = jax.random.split(rng, num_iters + 1)
    rng, *accept_keys = jax.random.split(rng, num_iters + 1)
    for i in tqdm_choice(range(num_iters), display_progressbar):
        lambdas_proposal_key, s_proposal_key = jax.random.split(proposal_keys[i], 2)
        lambdas_accept_key, s_accept_key = jax.random.split(proposal_keys[i], 2)

        current = jax.tree_util.tree_map(lambda val: val[:, i, :], samples)

        proposal = current.copy()
        proposal, kin_energy_diff = lambdas_hmc_proposal(
            lambdas_proposal_key, gradient_vmap, proposal, 
            lambdas_step_size, lambdas_num_steps
        )

        current, accepts = accept_test(lambdas_accept_key, log_likelihood_ratio_vmap, current, proposal, kin_energy_diff)
        acceptances["lambdas"] = acceptances["lambdas"].at[:, i].set(accepts)

        proposal = current.copy()
        proposal = s_proposal(s_proposal_key, proposal, s_proposal_iters)

        updates, accepts = accept_test(s_accept_key, log_likelihood_ratio_vmap, current, proposal)
        samples = jax.tree_util.tree_map(lambda samp, update: samp.at[:, i+1, :].set(update), samples, updates)
        acceptances["s"] = acceptances["s"].at[:, i].set(accepts)

    init_value_dropped_samples = jax.tree_util.tree_map(
        lambda val: val[:, 1:, :], samples
    )
    return init_value_dropped_samples, acceptances

def get_init_s_values(rng, one_init_s, num_chains, rand_iters):
    init_s_values = jnp.zeros((num_chains, one_init_s.shape[0])).at[:, :].set(one_init_s)
    init_s_values = s_proposal(rng, {"s": init_s_values}, rand_iters)["s"]
    return init_s_values

def force_dp_suff_stat_correct_sum(dp_suff_stat, correct_sum):
    rounded_dp_suff_stat = jnp.array(dp_suff_stat).round()
    approx_forced = (rounded_dp_suff_stat / rounded_dp_suff_stat.sum() * correct_sum).round()
    approx_forced_order_indices = jnp.flip(jnp.argsort(approx_forced))
    sum_diff = correct_sum - approx_forced.sum()
    diff_sign = int(jnp.sign(sum_diff))
    diff_abs = int(jnp.abs(sum_diff))
    if diff_sign == 0: 
        return approx_forced.astype(int)
    else:
        for i in range(diff_abs):
            approx_forced = approx_forced.at[approx_forced_order_indices[i]].add(diff_sign)

    return approx_forced.astype(int)

def force_dp_suff_stat_non_negative_correct_sum(dp_suff_stat, correct_sum):
    dp_suff_stat = jnp.select([dp_suff_stat >= 0], [dp_suff_stat], 0)
    if len(dp_suff_stat.shape) > 1:
        for i in range(dp_suff_stat.shape[0]):
            dp_suff_stat = dp_suff_stat.at[i, :].set(force_dp_suff_stat_correct_sum(dp_suff_stat[i, :], correct_sum))
        return dp_suff_stat
    else:
        return force_dp_suff_stat_correct_sum(dp_suff_stat, correct_sum)

def get_init_values(rng, dp_suff_stat, init_s_iters, n, num_chains):
    init_lambdas_key, init_s_key = jax.random.split(rng, 2)

    filled_dp_suff_stat = jnp.concatenate((jnp.array(dp_suff_stat), jnp.array((n - dp_suff_stat.sum(),))))
    one_init_suff_stat = force_dp_suff_stat_non_negative_correct_sum(filled_dp_suff_stat, n)
    init_s = get_init_s_values(init_s_key, one_init_suff_stat, num_chains, init_s_iters)
    init_s = force_dp_suff_stat_non_negative_correct_sum(init_s, n)

    init_lambdas = jax.random.normal(init_lambdas_key, (num_chains, dp_suff_stat.shape[0]))
    init_value = {"lambdas": init_lambdas, "s": init_s}
    return init_value



def unmarginalised_full_data_log_noise_pdf(params, dp_suff_stat, sigma_dp):
    suff_stat = params["s"]
    return dist.MultivariateNormal(suff_stat, sigma_dp**2 * jnp.eye(suff_stat.shape[0])).log_prob(dp_suff_stat)
    
def unmarginalised_full_data_log_X_likelihood(params, med, n):
    suff_stat = params["s"]
    return jnp.dot(params["lambdas"], suff_stat) - n * med.lambda0(params["lambdas"])

def unmarginalised_full_data_log_prior(lambdas):
    dim = lambdas.shape[0]
    return dist.MultivariateNormal(jnp.zeros(dim), 100 * jnp.eye(dim)).log_prob(lambdas)

def metropolis_hastings_unmarginalised_full_data(
    rng, log_noise_pdf, log_X_likelihood, log_prior, med,
    init_value, lambdas_step_size, lambdas_num_steps, 
    num_iters, num_chains, display_progressbar=True
    ):
    n_data = init_value["X"].shape[1]

    log_lambdas_target = jax.jit(lambda current: log_X_likelihood(current) + log_prior(current["lambdas"]))
    log_lambdas_target_ratio_vmap = jax.jit(jax.vmap(
        lambda current, proposal: log_lambdas_target(proposal) - log_lambdas_target(current),
        (0, 0), 0
    ))
    log_noise_ratio_vmap = jax.jit(jax.vmap(
        lambda current, proposal: log_noise_pdf(proposal) - log_noise_pdf(current),
        (0, 0), 0
    ))

    lambdas_proposal_gradient = jax.grad(lambda lambdas, s: log_lambdas_target({"lambdas": lambdas, "s": s}))
    lambdas_proposal_gradient_vmap = jax.jit(jax.vmap(
        lambda params: lambdas_proposal_gradient(params["lambdas"], params["s"]),
        0, 0
    ))
    vmap_queries = jax.jit(jax.vmap(
        lambda X: med.flat_queries(X).sum(axis=0)
    ))

    acceptances = jax.tree_util.tree_map(lambda val: jnp.zeros((num_chains, num_iters)), init_value)
    init_value["s"] = vmap_queries(init_value["X"])
    samples = list(range(num_iters + 1))
    samples[0] = init_value

    rng, *proposal_keys = jax.random.split(rng, num_iters + 1)
    rng, *accept_keys = jax.random.split(rng, num_iters + 1)
    for i in tqdm_choice(range(num_iters), display_progressbar):
        lambdas_proposal_key, X_proposal_key = jax.random.split(proposal_keys[i], 2)
        lambdas_accept_key, X_accept_key = jax.random.split(proposal_keys[i], 2)

        current = samples[i]

        proposal = current.copy()
        proposal, kin_energy_diff = lambdas_hmc_proposal(
            lambdas_proposal_key, lambdas_proposal_gradient_vmap, proposal, 
            lambdas_step_size, lambdas_num_steps
        )

        current, accepts = accept_test(lambdas_accept_key, log_lambdas_target_ratio_vmap, current, proposal, kin_energy_diff)
        acceptances["lambdas"] = acceptances["lambdas"].at[:, i].set(accepts)

        proposal = current.copy()
        proposal["X"] = jnp.zeros((init_value["X"].shape))
        for j in range(num_chains):
            X_proposal_key, key = jax.random.split(X_proposal_key, 2)
            X_sample = med.sample(key, current["lambdas"][j, :], n_sample=n_data)
            proposal["X"] = proposal["X"].at[j, :, :].set(X_sample)
        proposal["s"] = vmap_queries(proposal["X"])

        updates, accepts = accept_test(X_accept_key, log_noise_ratio_vmap, current, proposal)
        samples[i + 1] = updates
        acceptances["X"] = acceptances["X"].at[:, i].set(accepts)

    init_value_dropped_samples = samples[1:]
    return init_value_dropped_samples, acceptances