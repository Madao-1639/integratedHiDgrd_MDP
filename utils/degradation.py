import numpy as np
from mdp import OR_MDP_OneParam, My_MDP, My_MDP_Oracle

def get_deg_path(theta: float, sigma_square: float,
    t: int = 1, k_max: int = 400,
    n_sample: int = 1,) -> np.ndarray:
    '''
    Generate sample degradation paths with linear trend and brownian error term.

    Args:
    - theta (float): Slope parameter for the linear degradation trend
    - sigma_square (float): Volatility of brownian motion
    - t (int): Constant time interval between two consecutive observations
    - k_max (int): Maximum number of observation epochs
    - n_sample (int): Number of sample paths to generate

    Returns:
    - Generated sample paths (shape: (n_sample, k_max) if n_sample > 1 else (k_max,))
    '''
    shape = (n_sample,k_max) if n_sample > 1 else (k_max,)
    errInc = np.random.normal(loc=0,scale=sigma_square**0.5*t,size=shape)
    sample_path = (theta * t + errInc).cumsum(axis=-1)
    return sample_path

def get_rp_time(deg_path: np.ndarray, CL: int | np.ndarray) -> np.ndarray:
    '''
    Determine replacement time based on control limit.

    Args:
    - sample_deg_path (numpy.ndarray): Generated degradation paths.
    - CL (int or numpy.ndarray): Control limit for replacement decision.

    Returns:
    - Replacement time (numpy.ndarray): Time to replacement for each sample path.
    '''
    rp_path = deg_path >= CL
    rp_time = np.where(rp_path.any(axis=-1), rp_path.argmax(axis=-1) + 1, deg_path.shape[-1])
    return rp_time

def get_fl_time(deg_path: np.ndarray, mdp: My_MDP | None = None, threshold: float | None = None) -> np.ndarray:
    '''
    Determine failure time based on failure probability or threshold.

    Args:
    - sample_hi_path (numpy.ndarray): Generated high-impact degradation paths.
    - mdp (My_MDP or My_MDP_Oracle, optional): MDP instance to predict failure probabilities. Default None.
    - threshold (float, optional): Fixed threshold for failure decision. Default None.

    Returns:
    - Failure time (numpy.ndarray): Time to failure for each sample path.

    Notes:
    - Predicted failure occurs at the next epoch, so it is possible for failure time to exceed k_max.
    - When there is no failure within k_max epochs, failure time is set to k_max + 1. So forced replacement will not be considered as failure during evaluation. 
    '''
    if mdp is not None:
        hi_path = mdp.deg_model.l2hi(deg_path)
        fl_path = mdp.deg_model.predict_failure(hi_path)
        fl_path = np.random.random(fl_path.shape) < fl_path
        fl_time =  np.where(fl_path.any(axis=-1), fl_path.argmax(axis=-1) + 2, deg_path.shape[-1] + 1)
    elif threshold is not None:
        fl_path = deg_path >= threshold
        fl_time = np.where(fl_path.any(axis=-1), fl_path.argmax(axis=-1) + 1, deg_path.shape[-1] + 1)
    else:
        raise ValueError("Either mdp or threshold must be provided.")
    return fl_time

def eval_path(rp_time: np.ndarray, fl_time: np.ndarray,
    c1: float, c2: float, c3: float, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Evaluate total (discounted) cost and lifetime based on replacement and failure times.

    Args:
    - rp_time (numpy.ndarray): Replacement time for each sample path.
    - fl_time (numpy.ndarray): Failure time for each sample path.
    - c1 (float): Cost of preventive replacement.
    - c2 (float): Cost of reactive replacement.
    - c3 (float): Observation cost per time unit.
    - gamma (float): Discount factor. Default 0.99.

    Returns:
    - total_cost (numpy.ndarray): Total cost for each sample path.
    - lifetime (numpy.ndarray): Lifetime for each sample path.
    '''
    # Lifetime is minimum of replacement and failure time
    lifetime = np.fmin(rp_time, fl_time)
    # Calculate replacement cost (c1 if planned, c2 if failure)
    rp_cost = np.where(rp_time < fl_time, c1, c2)
    if 0 < gamma < 1:
        # Calculate total discounted cost
        disc_coef = np.power(gamma, lifetime - 1)
        total_disc_ob_cost = (1 - disc_coef) / (1 - gamma) * c3
        disc_rp_cost = disc_coef * (gamma * rp_cost)
        total_disc_cost = total_disc_ob_cost + disc_rp_cost
        return total_disc_cost, lifetime
    else:
        # Calculate total cost
        total_cost = (lifetime - 1) * c3 + rp_cost
        return total_cost, lifetime

def simulate(n_sample: int = 20000, n_theta: int = 3000, force_positive: bool = True,
            mu0: float = 0.05, sigma0_square: float = 0.01, sigma_square: float = 0.01,
            t : int = 1, k_max: int = 400,
            c1: float = 4, c2: float = 12, c3: float = 0.05, gamma: float = 0.99,
            mdp: My_MDP | None = None, mdp_PI: bool = True,
            or_mdp: OR_MDP_OneParam | None = None, or_mdp_PI: bool = True,
            oracle_mdp: My_MDP_Oracle | None = None, oracle_mdp_PI: bool = True,
            **mdp_kwargs):
    r'''
    Simulate cost, lifetime, and cost rate for My_MDP, OR_MDP_OneParam, and My_MDP_Oracle.

    Args:
    - n_sample: Number of degradation path samples per theta.
    - n_theta: Number of theta values to sample from prior distribution.
    - force_positive (bool, optional): If True, theta values are sampled from a normal distribution truncated at (0,+\infinity). Default True.
    - mu0 (float, optional): Prior mean of theta. Default 0.05.
    - sigma0_square (float, optional): Prior variance of theta. Default 0.01.
    - sigma_square (float, optional): Variance of the degradation noise. Default 0.01.
    - t (int, optional): Time step between two consecutive observations. Default 1.
    - k_max (int, optional): Maximum number of observation epochs. Default 400.
    - c1 (float, optional): Cost of preventive replacement. Default 4.
    - c2 (float, optional): Cost of reactive replacement. Default 12.
    - c3 (float, optional): Observation cost per time unit. Default 0.05.
    - gamma (float, optional): Discount factor for cost evaluation. Default 0.99.
    - mdp (My_MDP, optional) : If provided, use this MDP instance; otherwise initialize a new one with mdp_kwargs. Default None.
    - mdp_PI (bool, optional) : If True, compute the MDP policy with policy iteration; otherwise use value iteration. Default False.
    - or_mdp (OR_MDP_OneParam, optional) : If provided, use this oracle MDP instance; otherwise initialize a new one with mdp_kwargs. Default None.
    - or_mdp_PI (bool, optional) : If True, compute the oracle MDP policy with policy iteration; otherwise use value iteration. Default False.
    - oracle_mdp (My_MDP_Oracle, optional) : If provided, use this oracle MDP instance; otherwise initialize a new one with mdp_kwargs. Default None.
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
    - avg_cost_rate_list (numpy.ndarray): Estimated mean cost rate for the MDP policy for each sampled theta. (n_theta,)
    - or_avg_cost_list (numpy.ndarray): Estimated mean total cost (discounted if discounted_cost is True) for the oracle MDP (one-param) for each sampled theta. (n_theta,)
    - or_avg_lifetime_list (numpy.ndarray): Mean lifetime corresponding to or_avg_cost_list. (n_theta,)
    - or_avg_cost_rate_list (numpy.ndarray): Estimated mean cost rate for the oracle MDP (one-param) for each sampled theta. (n_theta,)
    - oracle_avg_cost_list (numpy.ndarray): Estimated mean total cost (discounted if discounted_cost is True) for the oracle MDP (theta known) for each sampled theta. (n_theta,)
    - oracle_avg_lifetime_list (numpy.ndarray): Mean lifetime corresponding to oracle_avg_cost_list. (n_theta,)
    - oracle_avg_cost_rate_list (numpy.ndarray): Estimated mean cost rate for the oracle MDP (theta known) for each sampled theta. (n_theta,)
    '''
    from scipy.stats import truncnorm

    if mdp is None:
        mdp = My_MDP(mu0 = mu0, sigma0_square = sigma0_square, sigma_square = sigma_square,
                    c1 = c1, c2 = c2, c3 = c3, gamma = gamma,
                    t = t, k_max = k_max, **mdp_kwargs)
    if or_mdp is None:
        or_mdp = OR_MDP_OneParam(mu0 = mu0, sigma0_square = sigma0_square, sigma_square = sigma_square,
                                c1 = c1, c2 = c2, c3 = c3, gamma = gamma,
                                t = t, k_max = k_max, **mdp_kwargs)
    if oracle_mdp is None:
        oracle_mdp = My_MDP_Oracle(theta = 0, sigma_square = sigma_square,
                                c1 = c1, c2 = c2, c3 = c3, gamma = gamma,
                                t = t, k_max = k_max, **mdp_kwargs)

    pi, *_ = mdp.policy_iteration(max_iter=100) if mdp_PI else mdp.value_iteration(max_iter=100, tol=1e-2)
    CL = mdp.l_index2l(pi)
    or_pi, *_ = or_mdp.policy_iteration(max_iter=100) if or_mdp_PI else or_mdp.value_iteration(max_iter=100, tol=1e-2)
    or_CL = or_mdp.l_index2l(or_pi)

    avg_cost_list, avg_lifetime_list, avg_cost_rate_list = np.empty(n_theta), np.empty(n_theta), np.empty(n_theta)
    or_avg_cost_list, or_avg_lifetime_list, or_avg_cost_rate_list = np.empty(n_theta), np.empty(n_theta), np.empty(n_theta)
    oracle_avg_cost_list, oracle_avg_lifetime_list, oracle_avg_cost_rate_list = np.empty(n_theta), np.empty(n_theta), np.empty(n_theta)

    if force_positive:
        theta_list = truncnorm.rvs(a = 0, b = np.inf, loc = mu0, scale = sigma0_square**0.5, size = n_theta)
    else:
        theta_list = np.random.normal(loc = mu0, scale = sigma0_square**0.5, size = n_theta)

    for i, theta in enumerate(theta_list):
        oracle_mdp.theta = theta
        oracle_pi, *_ = oracle_mdp.policy_iteration(max_iter=100) if oracle_mdp_PI else oracle_mdp.value_iteration(max_iter=100, tol=1e-2)
        oracle_CL = oracle_mdp.l_index2l(oracle_pi)

        sample_deg_path = get_deg_path(theta = theta, sigma_square = sigma_square,
            t = t, k_max = k_max, n_sample = n_sample)
        sample_fl_time = get_fl_time(sample_deg_path, mdp = mdp)

        sample_rp_time = get_rp_time(sample_deg_path, CL = CL)
        sample_cost, sample_lifetime = eval_path(rp_time = sample_rp_time, fl_time = sample_fl_time,
            c1 = c1, c2 = c2, c3 = c3, gamma = gamma)
        or_sample_rp_time = get_rp_time(sample_deg_path, CL = or_CL)
        or_sample_cost, or_sample_lifetime = eval_path(rp_time = or_sample_rp_time, fl_time = sample_fl_time,
            c1 = c1, c2 = c2, c3 = c3, gamma = gamma)
        oracle_sample_rp_time = get_rp_time(sample_deg_path, CL = oracle_CL)
        oracle_sample_cost, oracle_sample_lifetime = eval_path(rp_time = oracle_sample_rp_time, fl_time = sample_fl_time,
            c1 = c1, c2 = c2, c3 = c3, gamma = gamma)

        avg_cost, avg_lifetime, avg_cost_rate = sample_cost.mean(), sample_lifetime.mean(), (sample_cost/sample_lifetime).mean()
        or_avg_cost, or_avg_lifetime, or_avg_cost_rate = or_sample_cost.mean(), or_sample_lifetime.mean(), (or_sample_cost/or_sample_lifetime).mean()
        oracle_avg_cost, oracle_avg_lifetime, oracle_avg_cost_rate = oracle_sample_cost.mean(), oracle_sample_lifetime.mean(), (oracle_sample_cost/oracle_sample_lifetime).mean()
        avg_cost_list[i], avg_lifetime_list[i], avg_cost_rate_list[i] = avg_cost, avg_lifetime, avg_cost_rate
        or_avg_cost_list[i], or_avg_lifetime_list[i], or_avg_cost_rate_list[i] = or_avg_cost, or_avg_lifetime, or_avg_cost_rate
        oracle_avg_cost_list[i], oracle_avg_lifetime_list[i], oracle_avg_cost_rate_list[i] = oracle_avg_cost, oracle_avg_lifetime, oracle_avg_cost_rate

    return avg_cost_list, avg_lifetime_list, avg_cost_rate_list, \
        or_avg_cost_list, or_avg_lifetime_list, or_avg_cost_rate_list, \
        oracle_avg_cost_list , oracle_avg_lifetime_list, oracle_avg_cost_rate_list