from abc import ABC, abstractmethod
from enum import Enum, unique, auto
import numpy as np
from scipy.sparse import csr_array
from scipy.stats import norm
from .degradation import Linear, Exponential

@unique
class _Direction(Enum):
    '''Direction of last policy change during policy improvement.'''
    UP = auto()
    DOWN = auto()
    DONE = auto()



class BaseMDP_1D(ABC):
    """Base class for MDPs with 1-dimensional (l) state space."""
    def __init__(self, m: int,
        c1: float, c2: float, c3: float, gamma: float,
        t: int = 1,
        **deg_kwargs) -> None:
        # State space
        self.get_model(**deg_kwargs)
        self.m = m # Number of discretization intervals
        self.delta = self.deg_model.range / m # Discretization interval
        self.n_state = m + 1 # Number of states

        # Rewards (costs)
        self.c1 = c1 # Preventive replacement cost
        self.c2 = c2 # Reactive replacement cost
        self.c3 = c3 # Observation cost
        self.gamma = gamma # Discount factor

        # Transition Probabilities (remain to be defined in subclasses)
        pass

        # Other parameters
        self.t = t # The constant time between two consecutive observations

    def get_model(self, **deg_kwargs):
        self.deg_model = Linear(**deg_kwargs)

    def l2l_index(self, l):
        '''
        Discretize degradation signal (l) to degradation signal index (l_idx).
        l_idx = m represents failure state (l > threshold).
        l can be an array.
        '''
        return np.clip(l//self.delta - 1, a_min = 1 ,a_max = self.m)
    
    def l_index2l(self, l_idx):
        '''
        Remap degradation signal index (l_idx) to degradation signal (l).
        l_idx can be an array.
        '''
        return (1 + l_idx) * self.delta

    def state2index(self, l_idx):
        '''
        Map state (l_idx) to state index (l_idx + 1).
        State index 0 represents initial state.
        l_idx can be an array.
        '''
        return l_idx + 1

    def index2state(self, index):
        '''
        Remap state index (> 0) to state (l_idx).
        index can be an array.
        '''
        return index - 1

    @abstractmethod
    def gen_P_R(self, policy: int | np.ndarray, **kw_args) -> tuple[np.ndarray | csr_array, np.ndarray]:
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.

        Args:
        - policy (int or numpy.ndarray): Policy to be evaluated.
        - ...

        Returns:
        - P (scipy.sparse.csr_array or numpy.ndarray).
        - R (numpy.ndarray).
        '''
        raise NotImplementedError

    def policy_iteration(self, policy: int | None = None, max_iter: int = 10, **kw_args) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Perform policy iteration to compute an optimal policy for MDP.

        Args:
        - policy (int, optional): Initial policy to start iteration from. If None, set to `self.m` (always "do nothing"). Default None.
        - max_iter (int, optional):  Maximum number of policy iteration cycles to perform. Default 10. Iteration stops early if the policy becomes stable.
        - ...

        Returns:
        - policy (int).
        - V (numpy.ndarray): The value function associated with `policy`.
        - P (numpy.ndarray): Transition probability matrix returned by `self.policy_evaluation`.
        - R (numpy.ndarray): Reward vector returned by `self.policy_evaluation`.
        '''
        if policy is None:
            policy = self.m - 1
        _direction = None
        for _ in range(max_iter):
            V, P, R = self.policy_evaluation(policy, **kw_args)
            policy, _direction = self.policy_improvement(policy, V, P, R, _direction)
            if _direction == _Direction.DONE:
                break
        return policy, V, P, R

    def policy_evaluation(self, policy: int, **kw_args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Evaluate a given policy by solving the Bellman equation.
        
        Args:
        - policy (int).
        - ...

        Returns:
        - V (numpy.ndarray).
        - P (numpy.ndarray).
        - R (numpy.ndarray).
        '''
        P, R = self.gen_P_R(policy, **kw_args)
        V = np.linalg.solve(np.eye(self.n_state) - self.gamma * P, R)
        return V, P, R

    @abstractmethod
    def policy_improvement(self, policy: int, V: np.ndarray, P: np.ndarray, R: np.ndarray | None, _direction: _Direction| None = None) -> tuple[int, _Direction]:
        '''
        Improve a given policy based on the current value function.

        Args:
        - policy (int).
        - V (numpy.ndarray).
        - P (numpy.ndarray).
        - R (numpy.ndarray).
        - _direction (_Direction, optional): Direction of last change. Default None.

        Returns:
        - policy (int).
        - _direction (_Direction | None).
        '''
        raise NotImplementedError

    @abstractmethod
    def value_iteration(self, max_iter: int = 100, tol: float = 1e-3, **kw_args) -> tuple[int | np.ndarray, np.ndarray]:
        '''
        Perform synchronous value iteration to compute an optimal preventive-replacement policy.

        Args:
        - max_iter (int, optional):  Maximum number of value iteration cycles to perform. Default 100.
        - tol (float, optional):  Convergence tolerance for the value function. Default 1e-3.
        - ...

        Returns:
        - policy (int).
        - V (numpy.ndarray).
        '''
        raise NotImplementedError



class BaseMDP_2D(BaseMDP_1D, ABC):
    '''Base class for MDPs with 2-dimensional (k, lk) state space.'''
    def __init__(self, k_max: int, **kw_args) -> None:
        super().__init__(**kw_args)
        # State space
        self.k_max = k_max # Maximum observation epoch
        self.n_state = 1 + k_max * self.m  # Number of states

    def state2index(self, k_idx, lk_idx):
        '''
        Map state (k_idx, lk_idx) to state index.
        State index 0 represents initial state.
        k_idx and lk_idx can be arrays.
        '''
        return self.m  * k_idx + lk_idx + 1

    def index2state(self, index):
        '''
        Remap state index (> 0) to state (k_idx, lk_idx).
        '''
        return divmod(index - 1, self.m)

    def policy_iteration(self, policy: np.ndarray | None = None, max_iter: int = 10, **kw_args) -> tuple[np.ndarray, np.ndarray, csr_array, np.ndarray]:
        '''
        Perform policy iteration to compute an optimal policy for this MDP. This method runs iterative policy evaluation followed by policy improvement until the policy converges or a maximum number of iterations is reached.

        Args:
        - policy (numpy.ndarray, optional): Initial policy to start iteration from. Expected length is k_max. If None, every element is assigned to m - 1, except the final one set to 0 (a preventive replacement). Default None.
        - max_iter (int, optional): Maximum number of policy iteration cycles to perform. Default 10. Iteration stops early if the policy becomes stable.
        - ...

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`, as returned by `self.policy_evaluation`.
        - P (scipy.sparse.csr_array): Transition probability matrix returned by `self.policy_evaluation`.
        - R (numpy.ndarray): Reward vector returned by `self.policy_evaluation`.

        Notes:
        - Internally, a `_momentum` structure is passed to `self.policy_improvement` across iterations to accelerate updates by considering structured properties of optimal policy.

        References:
        - This algorithm follows Appendix A in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        if policy is None:
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max
            # policy = np.zeros(self.k_max,dtype=int)
        _momentum = [None] * (self.k_max - 1)
        for _ in range(max_iter):
            V, P, R = self.policy_evaluation(policy, **kw_args)
            policy, _momentum = self.policy_improvement(policy, V, P, R, _momentum)
            if all(_direction == _Direction.DONE for _direction in _momentum):
                break
        return policy, V, P, R

    def policy_evaluation(self, policy: np.ndarray, **kw_args) -> tuple[np.ndarray, csr_array, np.ndarray]:
        '''
        Evaluate a given policy by solving the Bellman equation.

        Args:
        - policy (numpy.ndarray).
        - ...

        Returns:
        - V (numpy.ndarray).
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).
        '''
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve
        P, R = self.gen_P_R(policy, **kw_args)
        V = spsolve(eye(self.n_state) - self.gamma * P, R)
        return V, P, R

    @abstractmethod
    def policy_improvement(self, policy: np.ndarray, V: np.ndarray, P: csr_array, R: np.ndarray | None, _momentum: list[_Direction | None]) -> tuple[np.ndarray, list[_Direction | None]]:
        '''
        Improve a given policy based on the current value function.

        Args:
        - policy (numpy.ndarray).
        - V (numpy.ndarray).
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).
        - _momentum (list of _Direction | None): Directions of last changes for each epoch.

        Returns:
        - policy (numpy.ndarray).
        - _momentum (list of _Direction | None).
        '''
        raise NotImplementedError



class OR_MDP(BaseMDP_2D):
    r'''
    Implementation for the literature
    > Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695.

    MDP of The Single-Unit Sensor-Based Replacement Problem (discretization scheme)
    - State space: W (k_max * (m + 1) + 1,)
        - One initial state (0,0)
        - Observation epochs (k = 1, ..., k_max) are represented by indices (0, ..., k_max - 1) for calculation convenience.
        - Range of degradation signals is shifted from [l_min, l_max] to [0,threshold], and discretized into m intervals. Note that there is also a failure interval, i.e. (threshold, +\infinity).
        - Intervals are represented by indices (0, ..., m) for calculation convenience.
        - Signals in interval i + 1 (index i) are estimated by (i + 1) * delta (upper endpoint)
    - Action space: A = {0: Do nothing, 1: Replace} (2,)
    - Transition Probabilities: P^{\pi} (k_max * (m + 1) + 1, k_max * (m + 1) + 1)
    - Rewards: R^{\pi} (k_max * (m + 1) + 1,)
    - Discount factor: gamma
    '''
    def __init__(self, mu0: float = -6.031, sigma0_square: float = 0.346, mu1: float = 8.061e-3, sigma1_square: float = 1.034e-5, sigma_square: float = 0.0073, **kw_args):
        # State space
        super().__init__(**kw_args)
        self.n_state = 1 + self.k_max * (self.m + 1) # Compared to BaseMDP_2D, OR_MDP has one more failure state per epoch
        self.threshold = self.deg_model.range 

        # Transition Probabilities
        self.mu0 = mu0 - self.deg_model.l_min
        self.sigma0_square = sigma0_square
        self.mu1 = mu1
        self.sigma1_square = sigma1_square
        self.sigma_square = sigma_square

    def state2index(self, k_idx, lk_idx):
        '''Considering the failure state, map function should be overwritten.'''
        return (self.m + 1)  * k_idx + lk_idx + 1

    def index2state(self, index):
        '''Considering the failure state, map function should be overwritten.'''
        return divmod(index - 1, self.m + 1)

    def gen_P_R(self, policy: np.ndarray, normalization: bool = True):
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.
        
        Args:
        - policy (array-like of int): Each element should be in the range [0, m - 1]. The policy determine whether to "do nothing" (l_k < l_k*) or "replace" (l_k >= l_k*) for each k.
        - normalization (bool, optional): If True, conditional probabilities over discretized next-level bins (given survival)are normalized to sum to 1. Default True.

        Returns:
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).

        Notes:
        - The method assumes that states are enumerated in a consistent flattened indexing scheme provided by `self.state2index`.
        - Numerical precision of the normal CDF and floating point summation may cause row sums to deviate from exactly 1; the `normalization` flag enables re-normalizing each row's survival-probability vector.
        - The implementation is optimized to pre-allocate nnz entries and fill CSR arrays incrementally; ensure nnz computation matches the policy and state-space sizes.
        '''
        # Generate R
            # Preventive replacement costs (c1, remain to be covered)
        R = self.c1 * np.ones(self.n_state)
            # Reactive replacement costs (c2)
        failure_indices = self.state2index(np.arange(self.k_max),self.m)
        R[failure_indices] = self.c2
            # Observation costs (c3)
        R[0] = self.c3
        repeated_k_indice = np.repeat(np.arange(self.k_max), policy)
        lk_indices = np.concatenate([np.arange(lk_star_index) for lk_star_index in policy])
        ob_indices = self.state2index(repeated_k_indice,lk_indices)
        R[ob_indices] = self.c3

        # Construct sparse P (CSR)
            # Calculate nnz (Number of Non-Zero entries)
        n_ob = (1 + np.sum(policy)) * (self.m + 1) # One initial state and states such that lk < lk* for all k (np.sum(policy))
        n_rp = np.sum((self.m + 1) - policy) # All states such that lk >= lk* (including failure state) for all k
        nnz = n_ob + n_rp
            # Pre-allocate necessary arrays
        csr_values, csr_row_indices, csr_col_indices = np.empty(nnz), np.empty(nnz, dtype=int), np.empty(nnz, dtype=int)

        # Calculate transition probabilities
            # Shared computation results
        lk_indices = np.arange(self.m + 1)
        L_vec = self.l_index2l(lk_indices[:-1]) # Covering range of lk
        L_lag_vec = self.l_index2l(lk_indices[:-1] - 1)
            # Initial state - do nothing
        # mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(0,0)
        # L_mu, L_sigma_square = self.gen_L_dist(0, mu1_, sigma1_square_)
        # L_mu, L_sigma_square = self.gen_L_dist(0, self.mu1, self.sigma1_square)
        L_mu = self.mu0 + self.mu1*self.t
        L_sigma = (self.sigma0_square + self.sigma1_square*self.t**2 + self.sigma_square)**0.5
        p_start = 0 # Pointer operation
        p_end = p_start + self.m
        csr_values[p_start:p_end] = (norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma))
        csr_values[p_end] = (1 - norm.cdf(self.threshold,loc=L_mu,scale=L_sigma))
        p_end += 1
        if normalization:
            csr_values[p_start:p_end] /= csr_values[p_start:p_end].sum()
        csr_row_indices[p_start:p_end] = 0
        csr_col_indices[p_start:p_end] = self.state2index(0,lk_indices)
            # k_idx = 0, ..., k_max - 1
        for k_idx in range(self.k_max):
            k = k_idx + 1
            for lk_idx in range(self.m + 1):
                p_start = p_end
                cur_state_index = self.state2index(k_idx, lk_idx)
                if lk_idx < policy[k_idx]: # Do nothing
                    lk = L_vec[lk_idx]
                    mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(k, lk)
                    L_mu, L_sigma_square = self.gen_L_dist(lk, mu1_, sigma1_square_)
                    L_sigma = L_sigma_square**0.5
                    p_end = p_start + self.m
                    csr_values[p_start:p_end] = (norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma))
                    csr_values[p_end] = (1 - norm.cdf(self.threshold,loc=L_mu,scale=L_sigma))
                    p_end += 1
                    if normalization:
                        csr_values[p_start:p_end] /= csr_values[p_start:p_end].sum()
                    csr_row_indices[p_start:p_end] = cur_state_index
                    csr_col_indices[p_start:p_end] = self.state2index(k,lk_indices)
                else: # Replace
                    csr_values[p_start] = 1.
                    csr_row_indices[p_start] = cur_state_index
                    csr_col_indices[p_start] = 0
                    p_end += 1
        P = csr_array((csr_values,(csr_row_indices, csr_col_indices)),shape=(self.n_state,self.n_state))
        return P, R

    def gen_posterior_dist(self, k, lk):
        '''
        Compute the posterior distributioins of parameters of 2-param Exponential Degradation Model with Brownian Error Terms.
        
        Args:
        - k (int).
        - lk (float).

        Returns:
        - mu0_ (float): Posterior mean for the first param.
        - mu1_ (float): Posterior mean for the second param.
        - sigma0_square_ (float): Posterior marginal variance for the first param.
        - sigma1_square_ (float): Posterior marginal variance for the second param.
        - rho_ (float): Posterior correlation coefficient between the two latent variables.
        
        References:
        - This update follows Proposition 1 in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        tmp1 = self.mu0*self.sigma_square*self.t # +self.l_min*self.sigma0_square
        tmp2 = self.sigma1_square*k*self.t+self.sigma_square
        tmp3 = lk*self.sigma1_square+self.mu1*self.sigma_square
        tmp4 = self.sigma0_square+self.t*self.sigma_square
        denominator = tmp4 * tmp2 - self.t*self.sigma0_square*self.sigma1_square
        mu0_ = (tmp1 * tmp2 - self.sigma0_square*self.t * tmp3) / denominator
        mu1_ =  (tmp3 * tmp4 - self.sigma1_square * tmp1) / denominator

        sigma0_square_ = self.sigma_square*self.sigma0_square*self.t*tmp2 / denominator
        sigma1_square_ = self.sigma_square*self.sigma1_square*tmp4 / denominator
        rho_ = -np.sqrt(self.sigma0_square*self.sigma1_square*self.t / (tmp4 * tmp2))
        return mu0_,mu1_,sigma0_square_,sigma1_square_,rho_

    def gen_L_dist(self, l, mu1_, sigma1_square_):
        '''
        Compute the Predictive distributioins of the next degradation signal (L) of Exponential Degradation Model with Brownian Error Terms after time interval t.
        
        Args:
        - l (float): Current observed degradation level (intercept).
        - mu1_ (float): Estimated mean for the second param.
        - sigma1_square_ (float): Estimated variance for the second param.

        Returns:
        - L_mu (float): Predictive mean of L.
        - L_sigma_square (float): Predictive variance of L.
        
        References:
        - This update follows Proposition 2 in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        L_mu = l + mu1_*self.t
        L_sigma_square = sigma1_square_*self.t**2 + self.sigma_square*self.t
        return L_mu, L_sigma_square

    def policy_improvement(self, policy: np.ndarray, V: np.ndarray, P: np.ndarray, R = None, _momentum: list[_Direction | None] = None) -> tuple[np.ndarray, list[_Direction | None]]:
        '''Keep R arg for compatibility, but not used.'''
        rp_cost = self.c1 + self.gamma * V[0]
        exp_trans_V = P@V
        for k_idx in range(self.k_max - 1):
            direction = _momentum[k_idx]
            if direction == _Direction.DONE:
                continue
            CL_index = policy[k_idx] # Index of Control Limit (CL)
            if direction != _Direction.DOWN and  CL_index < self.m - 1: # 'up' or None -> Try to shift CL upwards
                cur_state_index = self.state2index(k_idx, CL_index)
                ob_cost = self.c3 + self.gamma * exp_trans_V[cur_state_index]
                if ob_cost < rp_cost: # Shift
                    policy[k_idx] += 1
                    _momentum[k_idx] = _Direction.UP
                    continue
                elif direction == _Direction.UP:
                    _momentum[k_idx] = _Direction.DONE
                    continue
            if direction != _Direction.UP and CL_index > 0: # 'down' or None -> Try to shift CL downwards
                cur_state_index = self.state2index(k_idx, CL_index - 1)
                ob_cost = self.c3 + self.gamma * exp_trans_V[cur_state_index]
                if ob_cost > rp_cost: # Shift
                    policy[k_idx] -= 1
                    _momentum[k_idx] = _Direction.DOWN
                    continue
                elif direction == _Direction.DOWN:
                    _momentum[k_idx] = _Direction.DONE
                    continue
            _momentum[k_idx] = _Direction.DONE
        return policy, _momentum

    def value_iteration(self,max_iter: int = 100,tol: float = 1e-3, normalization: bool = True):
        '''
        Perform synchronous value iteration to compute an optimal preventive-replacement policy.

        Args:
        - max_iter (int, optional):  Maximum number of iterations to run the value-iteration loop. Iteration stops earlier if convergence (measured by the maximum absolute change in the value function) is achieved. Default 100.
        - tol (float, optional):  Convergence tolerance for the value function: stop when max_i |V_new[i] - V_old[i]| < tol. Default 1e-3.
        - normalization (bool, optional): If True, conditional probabilities over discretized next-level bins (given survival)are normalized to sum to 1. Default True.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`.
        '''
        V = np.zeros(self.n_state)
        lk_indices = np.arange(self.m + 1)
        L_vec = self.l_index2l(lk_indices[:-1])
        L_lag_vec = self.l_index2l(lk_indices[:-1] - 1)
        for _ in range(max_iter):
            V_pre = V.copy()
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max

            # Initial state - do nothing
            L_mu = self.mu0 + self.mu1*self.t
            L_sigma = (self.sigma0_square + self.sigma1_square*self.t**2 + self.sigma_square)**0.5
            trans_probs = np.empty(self.m + 1)
            trans_probs[:-1] = norm.cdf(L_vec, loc=L_mu, scale=L_sigma) - norm.cdf(L_lag_vec, loc=L_mu, scale=L_sigma)
            trans_probs[-1] = 1 - norm.cdf(self.threshold, loc=L_mu, scale=L_sigma)
            if normalization:
                trans_probs /= trans_probs.sum()
            next_indices = np.arange(1, self.m + 2)
            V[0] = self.c3 + self.gamma * trans_probs @ V[next_indices]

            # k_idx = 0, ..., k_max - 1
            prvt_rp_cost = self.c1 + self.gamma * V[0] # Preventive replacement cost
            rct_rp_cost = self.c2 + self.gamma * V[0] # Reactive replacement cost
            for k_idx in range(self.k_max):
                k = k_idx + 1
                for lk_idx in range(self.m):
                    # lk <= theshold
                    cur_state_index = self.state2index(k_idx, lk_idx)
                    if lk_idx >= policy[k_idx]:
                        # Preventive replacement
                        V[cur_state_index] = prvt_rp_cost
                    else:
                        # Decide whether to perform preventive replacement
                        lk = L_vec[lk_idx]
                        mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(k, lk)
                        L_mu, L_sigma_square = self.gen_L_dist(lk, mu1_, sigma1_square_)
                        L_sigma = L_sigma_square**0.5
                        trans_probs[:-1] = norm.cdf(L_vec, loc=L_mu, scale=L_sigma) - norm.cdf(L_lag_vec, loc=L_mu, scale=L_sigma)
                        trans_probs[-1] = 1 - norm.cdf(self.threshold, loc=L_mu, scale=L_sigma)
                        if normalization:
                            trans_probs /= trans_probs.sum()
                        next_indices = self.state2index(k, lk_indices)
                        ob_cost = self.c3 + self.gamma * trans_probs @ V[next_indices]
                        if ob_cost > prvt_rp_cost:
                            policy[k_idx] = lk_idx
                            V[cur_state_index] = prvt_rp_cost
                        else:
                            V[cur_state_index] = ob_cost
                # lk > theshold -> Reactive replacement
                cur_state_index = self.state2index(k_idx, self.m)
                V[cur_state_index] = rct_rp_cost
            if np.max(np.abs(V - V_pre)) < tol:
                break
        return policy, V



class OR_MDP_OneParam(OR_MDP):
    '''
    One parameter version of OR_MDP, which only retains the drift random-effect parameter (while discarding the random-effect offset) in exponential degradation model.
    '''
    def gen_posterior_dist(self, k, lk):
        '''
        Compute the posterior distributions of parameters for the 1-parameter Exponential Degradation Model with Brownian Error Terms. This model retains only the drift random-effect parameter (discarding the random-effect offset), resulting in a simplified parameter space compared to the parent class.
        
        Args:
        - k (int).
        - lk (float).

        Returns:
        - None: Placeholder for the posterior mean of the random-effect offset (omitted in the 1-parameter model).
        - mu1 (float): Posterior mean of the drift random-effect parameter.
        - None: Placeholder for the posterior variance of the random-effect offset (omitted in the 1-parameter model).
        - sigma1_square (float): Posterior variance of the drift random-effect parameter.
        - None: Placeholder for the posterior correlation coefficient between parameters (omitted in the 1-parameter model, as there is only one parameter).

        Notes:
        - The method maintains compatibility with the parent class `OR_MDP`'s `gen_posterior_dist` interface, which returns five values. Since the 1-parameter model omits the random-effect offset and only includes the drift parameter, the unused return positions are filled with `None` to preserve method signature consistency. This allows seamless integration with inherited methods (e.g., `gen_L_dist`) that expect the parent class's return structure.
        '''
        sigma1_square = 1/((1/self.sigma0_square)+(k*self.t/self.sigma_square))
        mu1 = sigma1_square*((self.mu0/self.sigma0_square)+(lk*self.t/self.sigma_square))
        return None, mu1, None, sigma1_square, None

    def gen_L_dist(self, l, mu1, sigma1_square):
        '''
        Compute the predictive distribution of the next degradation signal (L) for the 1-parameter Exponential Degradation Model with Brownian Error Terms after a time interval `t`. This method reuses the parent class's implementation while adapting to the simplified parameter space of the 1-parameter model.

        Args:
        - l (float): Current observed degradation level (intercept term).
        - mu1 (float): Posterior mean of the drift random-effect parameter (from the 1-parameter model).
        - sigma1_square (float): Posterior variance of the drift random-effect parameter (from the 1-parameter model).

        Returns:
        - L_mu (float): Predictive mean of the next degradation signal L.
        - L_sigma_square (float): Predictive variance of the next degradation signal L.

        Notes:
        - The method maintains compatibility with the parent class `OR_MDP`'s `gen_L_dist` interface by accepting parameters corresponding to the drift random-effect parameter and delegating computation to the parent class implementation. This ensures seamless integration with inherited logic (e.g., in `gen_P_R`) that relies on the parent class's return structure, while accommodating the simplified parameterization of the 1-parameter model.
        '''
        return super().gen_L_dist(l = l, mu1_= mu1, sigma1_square_ = sigma1_square)



class My_MDP(BaseMDP_2D):
    r'''
    MDP of The Single-Unit Sensor-Based Replacement Problem (discretization scheme)
    with failure probability: p = Sigmoid(hi), hi = -C1+exp(l-C2)
    - State space: W (k_max * m + 1,)
        - One initial state (0,0)
        - Observation epochs (k = 1, ..., k_max) are represented by indices (0, ..., k_max - 1) for calculation convenience.
        - Degradation signals are shifted to be nonnegtive and discretized into m intervals.
        - Intervals are represented by indices (0, ..., m - 1) for calculation convenience.
        - Signals in interval i + 1 (index i) are estimated by (i + 1) * delta (upper endpoint)
    - Action space: A = {0: Do nothing, 1: Replace} (2,)
    - Transition Probabilities: P^{\pi} (k_max * m + 1, k_max * m + 1)
    - Rewards: R^{\pi} (k_max * m + 1,)
    - Discount factor: gamma
    '''
    def __init__(self, mu0: float, sigma0_square: float, sigma_square: float,
        C1: float = 6.2, C2: float = -0.1, **kw_args):
        super().__init__(C1 = C1, C2 = C2 - kw_args['l_min'], **kw_args)

        # Transition Probabilities
        self.mu0 = mu0
        self.sigma0_square = sigma0_square
        self.sigma_square = sigma_square

    def get_model(self, **deg_kwargs):
        self.deg_model = Exponential(**deg_kwargs)
    def gen_P_R(self, policy: np.ndarray, normalization: bool = True):
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.
        
        Args:
        - policy (array-like of int): Each element should be in the range [0, m - 1]. The policy determine whether to "do nothing" (l_k < l_k*) or "preventive replace" (l_k >= l_k*) for each k.
        - normalization (bool, optional): If True, conditional probabilities over discretized next-level bins (given survival) are normalized to sum to 1. Default True.

        Returns:
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).

        Notes:
        - The method assumes that states are enumerated in a consistent flattened indexing scheme provided by `self.state2index`.
        - Numerical precision of the normal CDF and floating point summation may cause row sums to deviate from exactly 1; the `normalization` flag enables re-normalizing each row's survival-probability vector.
        - The implementation is optimized to pre-allocate nnz entries and fill CSR arrays incrementally; ensure nnz computation matches the policy and state-space sizes.
        '''
        # Generate R
            # Preventive replacement costs (c1, remain to be covered)
        R = self.c1 * np.ones(self.n_state)
            # Expected costs for "do nothing" (p*c2 + (1-p)*c3). Note that 0 < k < k_max
        repeated_k_indice = np.repeat(np.arange(self.k_max), policy)
        lk_indices = np.concatenate([np.arange(lk_star_index) for lk_star_index in policy])
        ob_indices = self.state2index(repeated_k_indice,lk_indices)
        lk = self.l_index2l(lk_indices)
        hi = self.deg_model.l2hi(lk)
        failure_prob_vec = self.deg_model.predict_failure(hi)
        R[ob_indices] = failure_prob_vec * self.c2 + (1 - failure_prob_vec) * self.c3
        failure_prob = self.deg_model.predict_failure(self.deg_model.l2hi(0))
        R[0] = failure_prob * self.c2 + (1 - failure_prob) * self.c3

        # Construct sparse P (CSR)
            # Calculate nnz (Number of Non-Zero entries)
        n_ob = (1 + np.sum(policy)) * (self.m + 1) # States such that lk < lk* for all k
        n_rp = np.sum(self.m - policy) # All states such that lk >= lk* for all k
        nnz = n_ob + n_rp
            # Pre-allocate necessary lists
        csr_values, csr_row_indices, csr_col_indices = np.empty(nnz), np.empty(nnz, dtype=int), np.empty(nnz, dtype=int)

        # Calculate transition probabilities
            # Shared computation results
        lk_indices = np.arange(self.m)
        L_vec = self.l_index2l(lk_indices) # Covering range of lk
        L_lag_vec = self.l_index2l(lk_indices - 1)
        hi_vec = self.deg_model.l2hi(L_vec)
        failure_prob_vec = self.deg_model.predict_failure(hi_vec)
            # Initial state - do nothing
        p_start = 0 # Pointer operation
        csr_values[p_start] = failure_prob # Calculated when generating R
        csr_col_indices[p_start] = 0
        L_mu, L_sigma_square = self.gen_L_dist(0,self.mu0,self.sigma0_square)
        L_sigma = L_sigma_square**0.5
        p_end = p_start + self.m + 1
        csr_row_indices[p_start:p_end] = 0
        p_start += 1
        evlv_p = norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma)
        if normalization:
            evlv_p /= evlv_p.sum()
        csr_values[p_start:p_end] = evlv_p * (1 - failure_prob)
        csr_col_indices[p_start:p_end] = self.state2index(0,lk_indices)
            # k_idx = 0, ..., k_max - 1
        for k_idx in range(self.k_max):
            k = k_idx + 1
            for lk_idx in range(self.m):
                p_start = p_end
                cur_state_index = self.state2index(k_idx, lk_idx)
                if lk_idx < policy[k_idx]: # Do nothing
                    failure_prob = failure_prob_vec[lk_idx]
                    csr_values[p_start] = failure_prob
                    csr_col_indices[p_start] = 0
                    lk = L_vec[lk_idx]
                    mu1,sigma1_square = self.gen_posterior_dist(k, lk)
                    L_mu, L_sigma_square = self.gen_L_dist(lk, mu1, sigma1_square)
                    L_sigma = L_sigma_square**0.5
                    p_end = p_start + self.m + 1
                    csr_row_indices[p_start:p_end] = cur_state_index
                    p_start += 1
                    evlv_p = norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma)
                    if normalization:
                        evlv_p /= evlv_p.sum()
                    csr_values[p_start:p_end] = evlv_p * (1 - failure_prob)
                    csr_col_indices[p_start:p_end] = self.state2index(k,lk_indices)
                else: # Replace
                    csr_values[p_start] = 1.
                    csr_row_indices[p_start] = cur_state_index
                    csr_col_indices[p_start] = 0
                    p_end += 1
        P = csr_array((csr_values,(csr_row_indices, csr_col_indices)),shape=(self.n_state,self.n_state))
        return P, R

    def gen_posterior_dist(self, k, lk):
        '''
        Compute the posterior distributioins of parameters of 1-param Exponential Degradation Model with Brownian Error Terms.
        
        Args:
        - k (int).
        - lk (float).

        Returns:
        - mu1 (float): Posterior mean for the param.
        - sigma1_square (float): Posterior variance for the param.
        '''
        sigma1_square = 1/((1/self.sigma0_square)+(k*self.t/self.sigma_square))
        mu1 = sigma1_square*((self.mu0/self.sigma0_square)+(lk*self.t/self.sigma_square))
        return mu1, sigma1_square

    def gen_L_dist(self, l, mu1, sigma1_square):
        '''
        Compute the Predictive distributioins of the next degradation signal (L) of Exponential Degradation Model with Brownian Error Terms after time interval t.
        
        Args:
        - l (float): Current observed degradation level (intercept).
        - mu1 (float): Estimated mean for the param.
        - sigma1_square (float): Estimated variance for the param.

        Returns:
        - L_mu (float): Predictive mean of L.
        - L_sigma_square (float): Predictive variance of L.
        '''
        L_mu = l + mu1*self.t
        L_sigma_square = sigma1_square*self.t**2 + self.sigma_square*self.t
        return L_mu, L_sigma_square

    def policy_improvement(self, policy: np.ndarray, V: np.ndarray, P: np.ndarray, R: np.ndarray, _momentum: _Direction| None = None) -> tuple[np.ndarray, list[_Direction | None]]:
        rp_cost = self.c1 + self.gamma * V[0]
        ob_costs = R + self.gamma * P@V
        for k_idx in range(self.k_max - 1):
            _direction = _momentum[k_idx]
            if _direction == _Direction.DONE:
                continue
            CL_index = policy[k_idx] # Index of Control Limit (CL)
            if _direction != _Direction.DOWN and  CL_index < self.m - 1: # Up or None -> Try to shift CL upwards
                cur_state_index = self.state2index(k_idx, CL_index)
                ob_cost = ob_costs[cur_state_index]
                if ob_cost < rp_cost: # Shift
                    policy[k_idx] += 1
                    _momentum[k_idx] = _Direction.UP
                    continue
                elif _direction == _Direction.UP:
                    _momentum[k_idx] = _Direction.DONE
                    continue
            if _direction != _Direction.UP and CL_index > 0: # Down or None -> Try to shift CL downwards
                cur_state_index = self.state2index(k_idx, CL_index - 1)
                ob_cost = ob_costs[cur_state_index]
                if ob_cost > rp_cost: # Shift
                    policy[k_idx] -= 1
                    _momentum[k_idx] = _Direction.DOWN
                    continue
                elif _direction == _Direction.DOWN:
                    _momentum[k_idx] = _Direction.DONE
                    continue
            _momentum[k_idx] = _Direction.DONE
        return policy, _momentum

    def value_iteration(self,max_iter: int = 100,tol: float = 1e-3, normalization: bool = True):
        '''
        Perform synchronous value iteration to compute an optimal preventive-replacement policy.

        Args:
        - max_iter (int, optional):  Maximum number of iterations to run the value-iteration loop. Iteration stops earlier if convergence (measured by the maximum absolute change in the value function) is achieved. Default 100.
        - tol (float, optional):  Convergence tolerance for the value function: stop when max_i |V_new[i] - V_old[i]| < tol. Default 1e-3.
        - normalization (bool, optional): If True, conditional probabilities over discretized next-level bins (given survival) are normalized to sum to 1. Default True.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`.
        '''
        V = np.zeros(self.n_state)
        trans_probs = np.empty(self.m + 1)
        next_indices = np.empty(self.m + 1, dtype=int)
        next_indices[0] = 0
        # Shared computation results
        lk_indices = np.arange(self.m)
        L_vec = self.l_index2l(lk_indices)
        L_lag_vec = self.l_index2l(lk_indices - 1)
        hi_vec = self.deg_model.l2hi(L_vec)
        failure_prob_vec = self.deg_model.predict_failure(hi_vec)
        for _ in range(max_iter):
            V_pre = V.copy()
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max

            # Initial state - do nothing
            failure_prob = self.deg_model.predict_failure(self.deg_model.l2hi(0))
            trans_probs[0] = failure_prob
            L_mu, L_sigma_square = self.gen_L_dist(0,self.mu0,self.sigma0_square)
            L_sigma = L_sigma_square**0.5
            evlv_p = norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma)
            if normalization:
                evlv_p /= evlv_p.sum()
            trans_probs[1:] = evlv_p * (1 - failure_prob)
            next_indices[1:] = self.state2index(0,lk_indices)
            V[0] = failure_prob * self.c2 + (1 - failure_prob) * self.c3 + self.gamma * trans_probs @ V[next_indices]

            # k_idx = 0, ..., k_max - 1
            rp_cost = self.c1 + self.gamma * V[0] # (Preventive) replacement cost
            for k_idx in range(self.k_max):
                k = k_idx + 1
                for lk_idx in range(self.m):
                    # lk <= CL
                    cur_state_index = self.state2index(k_idx, lk_idx)
                    if lk_idx >= policy[k_idx]:
                        # Preventive replacement
                        V[cur_state_index] = rp_cost
                    else:
                        # Decide whether to perform preventive replacement
                        failure_prob = failure_prob_vec[lk_idx]
                        trans_probs[0] = failure_prob
                        lk = L_vec[lk_idx]
                        mu1,sigma1_square = self.gen_posterior_dist(k, lk)
                        L_mu, L_sigma_square = self.gen_L_dist(lk, mu1, sigma1_square)
                        L_sigma = L_sigma_square**0.5
                        evlv_p = norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma)
                        if normalization:
                            evlv_p /= evlv_p.sum()
                        trans_probs[1:] = evlv_p * (1 - failure_prob)
                        next_indices[1:] = self.state2index(k, lk_indices)
                        ob_cost = failure_prob * self.c2 + (1 - failure_prob) * self.c3 + self.gamma * trans_probs @ V[next_indices]
                        if ob_cost > rp_cost:
                            policy[k_idx] = lk_idx
                            V[cur_state_index] = rp_cost
                        else:
                            V[cur_state_index] = ob_cost
            if np.max(np.abs(V - V_pre)) < tol:
                break
        return policy, V



class My_MDP_Oracle(BaseMDP_1D):
    '''
    An "oracle" version of `My_MDP` that knows the latent drift parameter theta.
    '''
    def __init__(self, theta: float, sigma_square: float, C1: float = 6.2, C2: float = -0.1, **kw_args):
        # Shared initialization
        super().__init__(**kw_args)

        # Transition Probabilities
        self.theta = theta
        self.sigma_square = sigma_square

        # Other parameters
        self.C1 = C1
        self.C2 = C2 - self.l_min

    def gen_P_R(self, policy: int, normalization: bool = True) -> tuple[np.ndarray, np.ndarray]:
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.
        
        Args:
        - policy (int): Control Limit (CL), which should be in the range [0, m - 1] and determines whether to "do nothing" (l < CL) or "preventive replace" (l >= CL).
        - normalization (bool, optional): If True, conditional probabilities over discretized next-level bins (given survival) are normalized to sum to 1. Default True.

        Returns:
        - P (numpy.ndarray).
        - R (numpy.ndarray).

        Notes:
        - Numerical precision of the normal CDF and floating point summation may cause row sums to deviate from exactly 1; the `normalization` flag enables re-normalizing each row's survival-probability vector.
        '''
        # Shared computation results
        l_indices = np.arange(self.m)
        L_vec = self.l_index2l(l_indices)
        L_lag_vec = self.l_index2l(l_indices - 1)
        ob_l_bool = l_indices < policy
        ob_l_vec = np.concatenate(([0], L_vec[ob_l_bool])) # Prepend initial state (l = 0)
        ob_hi_vec = self.deg_model.l2hi(ob_l_vec)
        failure_prob_vec = self.deg_model.predict_failure(ob_hi_vec)
        L_mu_vec, L_sigma_square = self.gen_L_dist(ob_l_vec)
        L_sigma = L_sigma_square**0.5

        # Generate R
        R = np.empty(self.n_state)
            # Expected costs for "do nothing" (p*c2 + (1-p)*c3)
        ob_state_bool = np.concatenate(([True], ob_l_bool)) # Prepend initial state
        R[ob_state_bool] = failure_prob_vec * (self.c2 - self.c3) + self.c3
            # Preventive replacement costs
        rp_state_bool = ~ob_state_bool
        R[rp_state_bool] = self.c1

        # Generate P
        P = np.zeros((self.n_state, self.n_state))
            # Initial state - do nothing
        P[ob_state_bool,0] = failure_prob_vec
            # l_idx = 1, ..., m
        evlv_p = np.vstack([norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma) for L_mu in L_mu_vec])
        if normalization:
            evlv_p /= evlv_p.sum(axis=1, keepdims=True)
        P[ob_state_bool,1:] = evlv_p * (1 - failure_prob_vec).reshape(-1,1) # Probabilities of l evolves to L and does not fail
        P[rp_state_bool,0] = 1
        return P, R

    def gen_L_dist(self, l):
        L_mu = l + self.theta * self.t
        L_sigma_square = self.sigma_square*self.t
        return L_mu, L_sigma_square

    def policy_improvement(self, policy: int, V: np.ndarray, P: np.ndarray, R: np.ndarray, _direction: _Direction | None = None):
        rp_cost = self.c1 + self.gamma * V[0]
        ob_costs = R + self.gamma * P@V
        if _direction != _Direction.DOWN and policy < self.m - 1: # Up or None -> Try to shift CL upwards
            ob_cost = ob_costs[policy]
            if ob_cost < rp_cost: # Shift
                policy += 1
                _direction = _Direction.UP
                return policy, _direction
            elif _direction == _Direction.UP:
                _direction = _Direction.DONE
                return policy, _direction
        if _direction != _Direction.UP and policy > 0: # Down or None -> Try to shift CL downwards
            ob_cost = ob_costs[policy-1]
            if ob_cost > rp_cost: # Shift
                policy -= 1
                _direction = _Direction.DOWN
                return policy, _direction
            elif _direction == _Direction.DOWN:
                _direction = _Direction.DONE
                return policy, _direction
        _direction = _Direction.DONE
        return policy, _direction

    def value_iteration(self, max_iter: int = 100, tol: float = 1e-3, normalization: bool = True):
        V = np.zeros(self.n_state)
        # Shared computation results
        l_indices = np.arange(self.m)
        L_vec = self.l_index2l(l_indices)
        L_lag_vec = self.l_index2l(l_indices - 1)
        l_vec = np.concatenate(([0],L_vec)) # Prepend initial state (l = 0)
        hi_vec = self.deg_model.l2hi(l_vec)
        failure_prob_vec = self.deg_model.predict_failure(hi_vec)
        L_mu_vec, L_sigma_square = self.gen_L_dist(l_vec)
        L_sigma = L_sigma_square**0.5
        evlv_p_arr = np.vstack([norm.cdf(L_vec,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag_vec,loc=L_mu,scale=L_sigma) for L_mu in L_mu_vec])
        if normalization:
            evlv_p_arr /= evlv_p_arr.sum(axis=1, keepdims=True)
        evlv_p_arr = evlv_p_arr * (1 - failure_prob_vec).reshape(-1,1) # Probabilities of l evolves to L and does not fail.
        for _ in range(max_iter):
            V_pre = V.copy()
            CL_state_idx = self.state2index(self.m - 1)

            rp_cost = self.c1 + self.gamma * V[0]
            for state_idx in range(self.n_state):
                # l <= CL
                if state_idx >= CL_state_idx:
                    # Preventive replacement
                    V[state_idx] = self.c1 + self.gamma * V[0]
                else:
                    # Decide whether to perform preventive replacement
                    failure_prob = failure_prob_vec[state_idx]
                    ob_cost = failure_prob * self.c2 + (1 - failure_prob) * self.c3 + \
                        self.gamma * (failure_prob * V[0] + evlv_p_arr[state_idx] @ V[1:])
                    if state_idx != 0 and ob_cost > rp_cost:
                        CL_state_idx = state_idx
                        V[state_idx:] = rp_cost
                        break
                    else:
                        V[state_idx] = ob_cost
            if np.max(np.abs(V - V_pre)) < tol:
                break
        policy = self.index2state(CL_state_idx)
        return policy, V