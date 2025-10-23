import numpy as np



class OR_MDP:
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
    def __init__(self, k_max: int = 2500, l_min: float = np.log(0.001), l_max: float = np.log(0.025), m: int = 20,
        c1: float = 3., c2: float = 12., c3: float = 0.005, gamma: float = 0.99,
        mu0: float = -6.031, sigma0_square: float = 0.346,
        mu1: float = 8.061e-3, sigma1_square: float = 1.034e-5,
        sigma_square: float = 0.0073,
        t: int = 2):
        # State space
        self.k_max = k_max
        self.l_min = l_min
        self.l_max = l_max
        self.threshold = l_max - l_min
        self.m = m
        self.delta = self.threshold / m
        self.n_states = 1 + k_max * (m + 1)

        # Rewards (costs)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.gamma = gamma

        # Transition Probabilities
        self.mu0 = mu0 - l_min
        self.sigma0_square = sigma0_square
        self.mu1 = mu1
        self.sigma1_square = sigma1_square
        self.sigma_square = sigma_square

        self.t = t # The constant time between two consecutive observations

    def l2l_index(self,l):
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
        return (1+l_idx)*self.delta

    def state2index(self, k_idx, lk_idx):
        '''
        Map state (k_idx, lk_idx) to state index.
        State index 0 represents initial state.
        k_idx and lk_idx can be arrays.
        '''
        return (self.m + 1)  * k_idx + lk_idx + 1

    def index2state(self,index):
        '''
        Remap state index (> 0) to state (k_idx, lk_idx).
        '''
        return divmod(index - 1, self.m + 1)

    def gen_P_R(self,policy):
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.
        
        Args:
        - policy (array-like of int): Each element should be in the range [0, m - 1]. The policy determine whether to "do nothing" (l_k < l_k*) or "replace" (l_k >= l_k*) for each k.

        Returns:
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).

        Notes:
        - The method assumes that states are enumerated in a consistent flattened indexing scheme provided by `self.state2index`.
        - Numerical precision of the normal CDF and floating point summation may cause row sums to deviate from exactly 1; if strict stochasticity is required, a row-normalization post-processing step may be applied.
        - The implementation is optimized to pre-allocate nnz entries and fill CSR arrays incrementally; ensure nnz computation matches the policy and state-space sizes.
        '''
        from scipy.sparse import csr_array
        from scipy.stats import norm

        # Generate R
            # Preventive replacement costs (c1, remain to be covered)
        R = self.c1 * np.ones(self.n_states)
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
        n_ob = (1 + np.sum(policy)) * (self.m + 1) # One initial state and states such that lk < lk* for all k
        n_rp = np.sum((self.m + 1) - policy) # All states such that lk >= lk* (including failure state) for all k
        nnz = n_ob + n_rp
            # Pre-allocate necessary lists
        csr_values, csr_row_indices, csr_col_indices = np.empty(nnz), np.empty(nnz, dtype=int), np.empty(nnz, dtype=int)

        # Calculate transition probabilities
            # Initial state - do nothing
        # mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(0,0)
        # L_mu, L_sigma_square = self.gen_L_dist(0, mu1_, sigma1_square_)
        # L_mu, L_sigma_square = self.gen_L_dist(0, self.mu1, self.sigma1_square)
        L_mu = self.mu0 + self.mu1*self.t
        L_sigma = (self.sigma0_square + self.sigma1_square*self.t**2 + self.sigma_square)**0.5
        L = self.l_index2l(np.arange(self.m))
        L_lag = L - self.delta # Notice that `L` and `L_lag` are shared arrays covering range of lk
        p_start = 0 # Pointer operation
        p_end = p_start + self.m
        csr_values[p_start:p_end] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma))
        csr_values[p_end] = (1 - norm.cdf(self.threshold,loc=L_mu,scale=L_sigma))
        p_end += 1
        csr_row_indices[p_start:p_end] = 0
        csr_col_indices[p_start:p_end] = np.arange(1, self.m + 2)
            # k_idx = 0, ..., k_max - 1
        for k_idx in range(self.k_max):
            k = k_idx + 1
            for lk_idx in range(self.m + 1):
                p_start = p_end
                cur_state_index = self.state2index(k_idx, lk_idx)
                if lk_idx < policy[k_idx]: # Do nothing
                    lk = L[lk_idx]
                    mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(k, lk)
                    L_mu, L_sigma_square = self.gen_L_dist(lk, mu1_, sigma1_square_)
                    L_sigma = L_sigma_square**0.5
                    p_end = p_start + self.m
                    csr_values[p_start:p_end] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma))
                    csr_values[p_end] = (1 - norm.cdf(self.threshold,loc=L_mu,scale=L_sigma))
                    p_end += 1
                    csr_row_indices[p_start:p_end] = cur_state_index
                    csr_col_indices[p_start:p_end] = self.state2index(k,np.arange(self.m + 1))
                else: # Replace
                    csr_values[p_start] = 1.
                    csr_row_indices[p_start] = cur_state_index
                    csr_col_indices[p_start] = 0
                    p_end += 1
        P = csr_array((csr_values,(csr_row_indices, csr_col_indices)),shape=(self.n_states,self.n_states))
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

    def gen_L_dist(self,lk, mu1_, sigma1_square_):
        '''
        Compute the Predictive distributioins of the next degradation signal (L) of Exponential Degradation Model with Brownian Error Terms after time interval t.
        
        Args:
        - lk (float): Current observed degradation level (intercept).
        - mu1_ (float): Estimated mean for the second param.
        - sigma1_square_ (float): Estimated variance for the second param.

        Returns:
        - L_mu (float): Predictive mean of L.
        - L_sigma_square (float): Predictive variance of L.
        
        References:
        - This update follows Proposition 2 in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        L_mu = lk + mu1_*self.t
        L_sigma_square = sigma1_square_*self.t**2 + self.sigma_square*self.t
        return L_mu, L_sigma_square

    def policy_iteration(self,policy=None,max_iter=10):
        '''
        Perform policy iteration to compute an optimal policy for this MDP. This method runs iterative policy evaluation followed by policy improvement until the policy converges or a maximum number of iterations is reached.

        Args:
        - policy (array-like of int, optional): Initial policy to start iteration from. Expected length is k_max. If None (default), every element is assigned m - 1, except the final one which is set to 0 (a preventive replacement). 
        - max_iter (int, optional):  Maximum number of policy-iteration cycles to perform. Default 10. Iteration stops early if the policy becomes stable.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`, as returned by `self.policy_evaluation`.
        - P (scipy.sparse.csr_array): Transition probability matrix returned by `self.policy_evaluation`.
        - R (numpy.ndarray): Reward vector returned by `self.policy_evaluation`.

        Notes:
        - Internally, a `momentum` structure is passed to `self.policy_improvement` across iterations to accelerate updates by considering structured properties of optimal policy.

        References:
        - This algorithm follows Appendix A in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        if policy is None:
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max
            # policy = np.zeros(self.k_max,dtype=int)
        momentum = [None] * (self.k_max - 1)
        for _ in range(max_iter):
            policy_pre = policy.copy()
            V, P, R = self.policy_evaluation(policy)
            policy, momentum = self.policy_improvement(policy, V, P, momentum)
            if np.array_equal(policy,policy_pre):
                break
        return policy, V, P, R

    def policy_evaluation(self, policy):
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve
        P, R = self.gen_P_R(policy)
        V = spsolve(eye(self.n_states) - self.gamma * P, R)
        return V, P, R

    def policy_improvement(self, policy, V, P, momentum):
        rp_cost = self.c1 + self.gamma * V[0]
        exp_trans_V = P@V
        for k_idx in range(self.k_max - 1):
            direction = momentum[k_idx]
            if direction == 'done':
                continue
            CL_index = policy[k_idx] # Index of Control Limit (CL)
            if direction != 'down' and  CL_index < self.m - 1: # 'up' or None -> Try to shift CL upwards
                cur_state_index = self.state2index(k_idx, CL_index)
                ob_cost = self.c3 + self.gamma * exp_trans_V[cur_state_index]
                if ob_cost < rp_cost: # Shift
                    policy[k_idx] += 1
                    momentum[k_idx] = 'up'
                    continue
                elif direction == 'up':
                    momentum[k_idx] = 'done'
                    continue
            if direction != 'up' and CL_index > 0: # 'down' or None -> Try to shift CL downwards
                cur_state_index = self.state2index(k_idx, CL_index - 1)
                ob_cost = self.c3 + self.gamma * exp_trans_V[cur_state_index]
                if ob_cost > rp_cost: # Shift
                    policy[k_idx] -= 1
                    momentum[k_idx] = 'down'
                    continue
                elif direction == 'down':
                    momentum[k_idx] = 'done'
                    continue
            momentum[k_idx] = 'done'
        return policy, momentum

    def value_iteration(self,max_iter=100,tol=1e-3):
        '''
        Perform synchronous value iteration to compute an optimal preventive-replacement policy.

        Args:
        - max_iter (int, optional):  Maximum number of iterations to run the value-iteration loop. Iteration stops earlier if convergence (measured by the maximum absolute change in the value function) is achieved. Default 100.
        - tol (float, optional):  Convergence tolerance for the value function: stop when max_i |V_new[i] - V_old[i]| < tol. Default 1e-3.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`.
        '''
        from scipy.stats import norm
        V = np.zeros(self.n_states)
        lk_indices = np.arange(self.m + 1)
        L = self.l_index2l(np.arange(self.m))
        L_lag = L - self.delta
        for _ in range(max_iter):
            V_pre = V.copy()
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max

            # Initial state - do nothing
            L_mu = self.mu0 + self.mu1*self.t
            L_sigma = (self.sigma0_square + self.sigma1_square*self.t**2 + self.sigma_square)**0.5
            trans_probs = np.empty(self.m + 1)
            trans_probs[:-1] = norm.cdf(L, loc=L_mu, scale=L_sigma) - norm.cdf(L_lag, loc=L_mu, scale=L_sigma)
            trans_probs[-1] = 1 - norm.cdf(self.threshold, loc=L_mu, scale=L_sigma)
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
                        lk = L[lk_idx]
                        mu0_,mu1_,sigma0_square_,sigma1_square_,rho_ = self.gen_posterior_dist(k, lk)
                        L_mu, L_sigma_square = self.gen_L_dist(lk, mu1_, sigma1_square_)
                        L_sigma = L_sigma_square**0.5
                        trans_probs[:-1] = norm.cdf(L, loc=L_mu, scale=L_sigma) - norm.cdf(L_lag, loc=L_mu, scale=L_sigma)
                        trans_probs[-1] = 1 - norm.cdf(self.threshold, loc=L_mu, scale=L_sigma)
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


class My_MDP:
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
    def __init__(self, k_max: int, l_min: float, l_max: float, m: int,
        c1: float, c2: float, c3: float, gamma: float,
        mu0: float, sigma0_square: float, sigma_square: float,
        t: int = 1, C1: float = 6.2, C2: float = -0.1):
        # State space
        self.k_max = k_max
        self.l_min = l_min
        self.l_max = l_max
        self.m = m
        self.delta = (l_max - l_min) / m
        self.n_states = k_max * m + 1

        # Rewards
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.gamma = gamma

        # Transition Probabilities
        self.mu0 = mu0
        self.sigma0_square = sigma0_square
        self.sigma_square = sigma_square

        self.t = t # The constant time between two consecutive observations
        self.C1 = C1
        self.C2 = C2 - l_min

    def l2l_index(self,l):
        '''
        Discretize degradation signal (l) to degradation signal index (l_idx).
        l can be an array.
        '''
        return l//self.delta - 1
    
    def l_index2l(self, l_idx):
        '''
        Remap degradation signal index (l_idx) to degradation signal (l).
        l_idx can be an array.
        '''
        return (1+l_idx)*self.delta

    def state2index(self, k_idx, lk_idx):
        '''
        Map state (k_idx, lk_idx) to state index.
        State index 0 represents initial state.
        k_idx and lk_idx can be arrays.
        '''
        return self.m  * k_idx + lk_idx + 1

    def index2state(self,index):
        '''
        Remap state index (> 0) to state (k_idx, lk_idx).
        '''
        return divmod(index - 1, self.m)

    def gen_P_R(self,policy):
        '''
        Generate transition probability matrix P and reward vector R given a control limit policy.
        
        Args:
        - policy (array-like of int): Each element should be in the range [0, m - 1]. The policy determine whether to "do nothing" (l_k < l_k*) or "preventive replace" (l_k >= l_k*) for each k.

        Returns:
        - P (scipy.sparse.csr_array).
        - R (numpy.ndarray).

        Notes:
        - The method assumes that states are enumerated in a consistent flattened indexing scheme provided by `self.state2index`.
        - Numerical precision of the normal CDF and floating point summation may cause row sums to deviate from exactly 1; if strict stochasticity is required, a row-normalization post-processing step may be applied.
        - The implementation is optimized to pre-allocate nnz entries and fill CSR arrays incrementally; ensure nnz computation matches the policy and state-space sizes.
        '''
        from scipy.sparse import csr_array
        from scipy.stats import norm

        # Generate R
            # Preventive replacement costs (c1, remain to be covered)
        R = self.c1 * np.ones(self.n_states)
            # Expected costs for "do nothing" (p*c2 + (1-p)*c3). Notice that 0 < k < k_max
        repeated_k_indice = np.repeat(np.arange(self.k_max), policy)
        lk_indices = np.concatenate([np.arange(lk_star_index) for lk_star_index in policy])
        ob_indices = self.state2index(repeated_k_indice,lk_indices)
        failure_probs = self.predict_failure(self.l_index2l(lk_indices))
        R[ob_indices] = failure_probs * self.c2 + (1 - failure_probs) * self.c3
        failure_prob = self.predict_failure(0)
        R[0] = failure_prob * self.c2 + (1 - failure_prob) * self.c3

        # Construct sparse P (CSR)
            # Calculate nnz (Number of Non-Zero entries)
        n_ob = (1 + np.sum(policy)) * (self.m + 1) # States such that lk < lk* for all k
        n_rp = np.sum(self.m - policy) # All states such that lk >= lk* for all k
        nnz = n_ob + n_rp
            # Pre-allocate necessary lists
        csr_values, csr_row_indices, csr_col_indices = np.empty(nnz), np.empty(nnz, dtype=int), np.empty(nnz, dtype=int)

        # Calculate transition probabilities
            # Initial state - do nothing
        p_start = 0 # Pointer operation
        csr_values[p_start] = failure_prob # Calculated when generating R
        csr_col_indices[p_start] = 0
        L_mu, L_sigma_square = self.gen_L_dist(0,self.mu0,self.sigma0_square)
        L_sigma = L_sigma_square**0.5
        L = self.l_index2l(np.arange(self.m))
        L_lag = L - self.delta # Notice that `L` and `L_lag` are shared arrays covering range of lk
        p_end = p_start + self.m + 1
        csr_row_indices[p_start:p_end] = 0
        p_start += 1
        csr_values[p_start:p_end] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma)) * (1 - failure_prob)
        csr_col_indices[p_start:p_end] = np.arange(1, self.m + 1)
            # k_idx = 0, ..., k_max - 1
        failure_probs = self.predict_failure(L)
        for k_idx in range(self.k_max):
            k = k_idx + 1
            for lk_idx in range(self.m):
                p_start = p_end
                cur_state_index = self.state2index(k_idx, lk_idx)
                if lk_idx < policy[k_idx]: # Do nothing
                    failure_prob = failure_probs[lk_idx]
                    csr_values[p_start] = failure_prob
                    csr_col_indices[p_start] = 0
                    lk = L[lk_idx]
                    mu1,sigma1_square = self.gen_posterior_dist(k, lk)
                    L_mu, L_sigma_square = self.gen_L_dist(lk, mu1, sigma1_square)
                    L_sigma = L_sigma_square**0.5
                    p_end = p_start + self.m + 1
                    csr_row_indices[p_start:p_end] = cur_state_index
                    p_start += 1
                    csr_values[p_start:p_end] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma)) * (1 - failure_prob)
                    csr_col_indices[p_start:p_end] = self.state2index(k,np.arange(self.m))
                else: # Replace
                    csr_values[p_start] = 1.
                    csr_row_indices[p_start] = cur_state_index
                    csr_col_indices[p_start] = 0
                    p_end += 1
        P = csr_array((csr_values,(csr_row_indices, csr_col_indices)),shape=(self.n_states,self.n_states))
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

    def gen_L_dist(self,lk, mu1, sigma1_square):
        '''
        Compute the Predictive distributioins of the next degradation signal (L) of Exponential Degradation Model with Brownian Error Terms after time interval t.
        
        Args:
        - lk (float): Current observed degradation level (intercept).
        - mu1 (float): Estimated mean for the param.
        - sigma1_square (float): Estimated variance for the param.

        Returns:
        - L_mu (float): Predictive mean of L.
        - L_sigma_square (float): Predictive variance of L.
        '''
        L_mu = lk + mu1*self.t
        L_sigma_square = sigma1_square*self.t**2 + self.sigma_square*self.t
        return L_mu, L_sigma_square

    def predict_failure(self, l):
        '''
        Failure probability at next epoch.
        '''
        # Reverse log transformation
        hi = -self.C1 + np.exp(l - self.C2)
        # Sigmoid function
        p = np.where(
            hi >= 0,
            1 / (1 + np.exp(-hi)),
            np.exp(hi) / (1 + np.exp(hi))
        )
        return p

    def policy_iteration(self,policy=None,max_iter=10):
        '''
        Perform policy iteration to compute an optimal policy for this MDP. This method runs iterative policy evaluation followed by policy improvement until the policy converges or a maximum number of iterations is reached.

        Args:
        - policy (array-like of int, optional): Initial policy to start iteration from. Expected length is k_max. If None (default), every element is assigned m - 1, except the final one which is set to 0 (a preventive replacement). 
        - max_iter (int, optional):  Maximum number of policy-iteration cycles to perform. Default 10. Iteration stops early if the policy becomes stable.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`, as returned by `self.policy_evaluation`.
        - P (scipy.sparse.csr_array): Transition probability matrix returned by `self.policy_evaluation`.
        - R (numpy.ndarray): Reward vector returned by `self.policy_evaluation`.

        Notes:
        - Internally, a `momentum` structure is passed to `self.policy_improvement` across iterations to accelerate updates by considering structured properties of optimal policy.

        References:
        - This algorithm follows Appendix A in the literature "Alaa H. Elwany, Nagi Z. Gebraeel, Lisa M. Maillart, (2011) Structured Replacement Policies for Components with Complex Degradation Processes and Dedicated Sensors. Operations Research 59(3):684-695."
        '''
        if policy is None:
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max
            # policy = np.zeros(self.k_max,dtype=int)
        momentum = [None] * (self.k_max - 1)
        for _ in range(max_iter):
            policy_pre = policy.copy()
            V, P, R = self.policy_evaluation(policy)
            policy, momentum = self.policy_improvement(policy, V, P, R, momentum)
            if np.array_equal(policy,policy_pre):
                break
        return policy, V, P, R

    def policy_evaluation(self, policy):
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve
        P, R = self.gen_P_R(policy)
        V = spsolve(eye(self.n_states) - self.gamma * P, R)
        return V, P, R

    def policy_improvement(self, policy, V, P, R, momentum):
        rp_cost = self.c1 + self.gamma * V[0]
        ob_costs = R + self.gamma * P@V
        for k_idx in range(self.k_max - 1):
            direction = momentum[k_idx]
            if direction == 'done':
                continue
            CL_index = policy[k_idx] # Index of Control Limit (CL)
            if direction != 'down' and  CL_index < self.m - 1: # 'up' or None -> Try to shift CL upwards
                cur_state_index = self.state2index(k_idx, CL_index)
                ob_cost = ob_costs[cur_state_index]
                if ob_cost < rp_cost: # Shift
                    policy[k_idx] += 1
                    momentum[k_idx] = 'up'
                    continue
                elif direction == 'up':
                    momentum[k_idx] = 'done'
                    continue
            if direction != 'up' and CL_index > 0: # 'down' or None -> Try to shift CL downwards
                cur_state_index = self.state2index(k_idx, CL_index - 1)
                ob_cost = ob_costs[cur_state_index]
                if ob_cost > rp_cost: # Shift
                    policy[k_idx] -= 1
                    momentum[k_idx] = 'down'
                    continue
                elif direction == 'down':
                    momentum[k_idx] = 'done'
                    continue
            momentum[k_idx] = 'done'
        return policy, momentum

    def value_iteration(self,max_iter=100,tol=1e-3):
        '''
        Perform synchronous value iteration to compute an optimal preventive-replacement policy.

        Args:
        - max_iter (int, optional):  Maximum number of iterations to run the value-iteration loop. Iteration stops earlier if convergence (measured by the maximum absolute change in the value function) is achieved. Default 100.
        - tol (float, optional):  Convergence tolerance for the value function: stop when max_i |V_new[i] - V_old[i]| < tol. Default 1e-3.

        Returns:
        - policy (numpy.ndarray).
        - V (numpy.ndarray): The value function associated with `policy`.
        '''
        from scipy.stats import norm
        V = np.zeros(self.n_states)
        lk_indices = np.arange(self.m)
        L = self.l_index2l(lk_indices)
        L_lag = L - self.delta
        failure_probs = self.predict_failure(L)
        for _ in range(max_iter):
            V_pre = V.copy()
            policy = (self.m - 1) * np.ones(self.k_max,dtype=int)
            policy[-1] = 0 # Always perform a preventive replacement at k = k_max

            # Initial state - do nothing
            failure_prob = self.predict_failure(0)
            trans_probs = np.empty(self.m + 1)
            trans_probs[0] = failure_prob
            L_mu, L_sigma_square = self.gen_L_dist(0,self.mu0,self.sigma0_square)
            L_sigma = L_sigma_square**0.5
            trans_probs[1:] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma)) * (1 - failure_prob)
            next_indices = np.arange(self.m + 1)
            V[0] = failure_prob * self.c2 + (1 - failure_prob) * self.c3 + self.gamma * trans_probs @ V[next_indices]

            # k_idx = 0, ..., k_max - 1
            rp_cost = self.c1 + self.gamma * V[0] # (Preventive) replacement cost
            for k_idx in range(self.k_max):
                k = k_idx + 1
                for lk_idx in range(self.m):
                    # lk <= theshold
                    cur_state_index = self.state2index(k_idx, lk_idx)
                    if lk_idx >= policy[k_idx]:
                        # Preventive replacement
                        V[cur_state_index] = rp_cost
                    else:
                        # Decide whether to perform preventive replacement
                        failure_prob = failure_probs[lk_idx]
                        trans_probs[0] = failure_prob
                        lk = L[lk_idx]
                        mu1,sigma1_square = self.gen_posterior_dist(k, lk)
                        L_mu, L_sigma_square = self.gen_L_dist(lk, mu1, sigma1_square)
                        L_sigma = L_sigma_square**0.5
                        trans_probs[1:] = (norm.cdf(L,loc=L_mu,scale=L_sigma) - norm.cdf(L_lag,loc=L_mu,scale=L_sigma)) * (1 - failure_prob)
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