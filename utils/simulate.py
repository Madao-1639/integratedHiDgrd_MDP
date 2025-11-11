import numpy as np
from mdp import My_MDP, My_MDP_Oracle


def get_deg_path(theta: float, sigma_square: float,
    t: int = 1, k_max: int = 400,
    n_sample: int = 1,):
    '''
    Generate sample degradation paths with linear trend and brownian error term.

    Args:
    - theta: Slope parameter for the linear degradation trend
    - sigma_square: Volatility of brownian motion
    - t: Constant time interval between two consecutive observations
    - k_max: Maximum number of observation epochs
    - n_sample: Number of sample paths to generate

    Returns:
    - Generated sample paths (shape: (n_sample, k_max) if n_sample > 1 else (k_max,))
    '''
    shape = (n_sample,k_max) if n_sample > 1 else (k_max,)
    errInc = np.random.normal(loc=0,scale=sigma_square**0.5*t,size=shape)
    sample_path = (theta * t + errInc).cumsum(axis=-1)
    return sample_path

def eval_path(sample_deg_path, mdp: My_MDP | My_MDP_Oracle, CL,
    c1: float, c2: float, c3: float, gamma: float = 0.99, discounted_cost: bool = True):
    '''
    Evaluate total discounted cost and lifetime for given degradation paths.

    Args:
    - sample_paths: Generated degradation paths
    - mdp: MDP or Oracle MDP instance
    - CL: Control limit for replacement decision
    - c1: Cost of preventive replacement
    - c2: Cost of failure replacement
    - c3: Observation cost
    - gamma: Discount factor
    - discounted_cost (bool, optional) : If True, calculate total discounted cost. Default True.

    Returns:
    - Total (discounted) cost for each sample path
    - Lifetime (time to replacement/failure) for each sample path
    '''
    # Determine replacement time (first crossing of control limit, regardless of failure)
    rp_path = sample_deg_path >= CL
    rp_time = np.where(rp_path.any(axis=-1), rp_path.argmax(axis=-1) + 1, mdp.k_max)
    # Determine failure time (first occurrence of failure)
    failure_path = mdp.predict_failure(sample_deg_path)
    failure_path = np.random.random(failure_path.shape) < failure_path
    fl_time =  np.where(failure_path.any(axis=-1), failure_path.argmax(axis=-1) + 1, mdp.k_max)
    # Lifetime is minimum of replacement and failure time
    lifetime = np.fmin(rp_time, fl_time)
    # Calculate replacement cost (c1 if planned, c2 if failure)
    rp_cost = np.where(rp_time < fl_time, c1, c2)
    if discounted_cost:
        # Calculate total discounted cost
        disc_coef = np.power(gamma,lifetime - 1)
        total_disc_ob_cost = (1-disc_coef)/(1-gamma) * c3
        disc_rp_cost = disc_coef * (gamma * rp_cost)
        total_disc_cost = total_disc_ob_cost + disc_rp_cost
        return total_disc_cost, lifetime
    else:
        # Calculate total cost
        total_cost = (lifetime - 1) * c3 + rp_cost
        return total_cost, lifetime

def simulate(n_sample: int = 20000, n_theta: int = 3000,
            force_positive: bool = True, mdp_PI: bool = False, oracle_mdp_PI: bool = True,
            discounted_cost: bool = True, **mdp_kwargs):
    r'''
    Simulate expected cost and lifetime for MDP and Oracle MDP.

    Args:
    - n_sample: Number of degradation path samples per theta
    - n_theta: Number of theta values to sample from prior distribution
    - force_positive (bool, optional): If True, theta values are sampled from a normal distribution truncated at (0,+\infinity). Default True.
    - mdp_PI (bool, optional) : If True, compute the MDP policy with policy iteration; otherwise use value iteration. Default False.
    - oracle_mdp_PI (bool, optional) : If True, compute the oracle MDP policy with policy iteration; otherwise use value iteration. Default True.
    - discounted_cost (bool, optional) : If True, evaluate total discounted cost on simulated trajectories. Default True.
    - ** mdp_kwargs: Keyword arguments for MDP initialization, including:
        - mu0, sigma0_square: Prior distribution parameters for theta
        - sigma_square: Noise variance
        - t: Time step
        - k_max: Maximum time steps
        - c1, c2, c3: Cost parameters
        - gamma: Discount factor
        - ...

    Returns:
    - avg_cost_list (numpy.ndarray): Estimated mean total cost (discounted if discounted_cost == True) for the MDP policy for each sampled theta. (n_theta,)
    - avg_lifetime_list (numpy.ndarray): Mean lifetime corresponding to avg_cost_list. (n_theta,)
    - oracle_avg_cost_list (numpy.ndarray): Estimated mean total cost (discounted if discounted_cost is True) for the oracle MDP (theta known) for each sampled theta. (n_theta,)
    - oracle_avg_lifetime_list (numpy.ndarray): Mean lifetime corresponding to oracle_avg_cost_list. (n_theta,)
    '''
    from scipy.stats import truncnorm
    mdp = My_MDP(**mdp_kwargs)
    pi, *_ = mdp.policy_iteration(max_iter=100) if mdp_PI else mdp.value_iteration(max_iter=100, tol=1e-2)
    CL = mdp.l_index2l(pi)

    avg_cost_list, avg_lifetime_list = np.empty(n_theta), np.empty(n_theta)
    oracle_avg_cost_list, oracle_avg_lifetime_list = np.empty(n_theta), np.empty(n_theta)
    if force_positive:
        theta_list = truncnorm.rvs(a = 0, b = np.inf, loc = mdp_kwargs['mu0'], scale = mdp_kwargs['sigma0_square']**0.5, size = n_theta)
    else:
        theta_list = np.random.normal(loc = mdp_kwargs['mu0'], scale = mdp_kwargs['sigma0_square']**0.5, size = n_theta)
    del mdp_kwargs['mu0'], mdp_kwargs['sigma0_square']

    for i in range(n_theta):
        theta = theta_list[i]
        oracle_mdp = My_MDP_Oracle(theta = theta, **mdp_kwargs)
        oracle_pi, *_ = oracle_mdp.policy_iteration(max_iter=100) if oracle_mdp_PI else oracle_mdp.value_iteration(max_iter=100, tol=1e-2)
        oracle_CL = oracle_mdp.l_index2l(oracle_pi)

        sample_deg_path = get_deg_path(theta = theta, sigma_square = mdp_kwargs['sigma_square'],
            t = mdp.t, k_max = mdp_kwargs['k_max'], n_sample = n_sample)

        sample_cost, sample_lifetime = eval_path(sample_deg_path, mdp = mdp, CL = CL,
            c1 = mdp_kwargs['c1'], c2 = mdp_kwargs['c2'], c3 = mdp_kwargs['c3'], gamma = mdp_kwargs['gamma'], discounted_cost = discounted_cost)
        oracle_sample_cost, oracle_sample_lifetime = eval_path(sample_deg_path, mdp = oracle_mdp, CL = oracle_CL,
            c1 = mdp_kwargs['c1'], c2 = mdp_kwargs['c2'], c3 = mdp_kwargs['c3'], gamma = mdp_kwargs['gamma'], discounted_cost = discounted_cost)
        avg_cost, avg_lifetime = sample_cost.mean(), sample_lifetime.mean()
        oracle_avg_cost, oracle_avg_lifetime = oracle_sample_cost.mean(), oracle_sample_lifetime.mean()
        avg_cost_list[i], avg_lifetime_list[i] = avg_cost, avg_lifetime
        oracle_avg_cost_list[i], oracle_avg_lifetime_list[i] = oracle_avg_cost, oracle_avg_lifetime

    return avg_cost_list, avg_lifetime_list, \
        oracle_avg_cost_list , oracle_avg_lifetime_list