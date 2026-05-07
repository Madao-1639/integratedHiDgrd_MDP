import numpy as np


def _sigmoid(x):
    x = x.clip(-100, 100) # Avoid overflow
    # Sigmoid function
    p = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
    return p





class Linear:
    r'''
    Linear transformation to scale HI to [0, +\infinity).
    hi = l + l_min <==> l = hi - l_min
    '''
    def __init__(self, l_min: float = 0.0, l_max: float = 5.0) -> None:
        # State space
        self.l_min = l_min # Minimum degradation signal
        self.l_max = l_max # Maximum degradation signal

    @property
    def range(self):
        return self.l_max - self.l_min

    def hi2l(self,hi):
        r'''
        Transform Health Index (hi \in (-\infinity, +\infinity)) to degradation signal (l \in [0, +\infinity)).
        hi can be an array.
        '''
        return hi - self.l_min

    def l2hi(self,l):
        r'''
        Restore degradation signal (l \in [0, +\infinity)) to Health Index (hi \in (-\infinity, +\infinity)).
        l can be an array.
        '''
        return l + self.l_min



class Exponential(Linear):
    r'''
    Logarithmic transformation.
    hi = exp(l - C2) - C1 <==> l = log(hi + C1) + C2
    p = sigmoid(hi) = 1 / (1 + exp(-hi))
    '''
    def __init__(self, C1: float = 1.0, C2: float = 0.0, **kw_args) -> None:
        # State space
        super().__init__(**kw_args)
        self.C1 = C1 # Shift parameter in the logarithmic transformation
        self.C2 = C2 # Offset for the degradation signal

    def hi2l(self,hi):
        r'''
        Transform from Health Index (hi \in (-\infinity, +\infinity)) to degradation signal (l \in [0, +\infinity)).
        hi can be an array.
        '''
        return np.log(hi + self.C1) + self.C2

    def l2hi(self,l):
        r'''
        Inverse transform from degradation signal (l \in [0, +\infinity)) to Health Index (hi \in (-\infinity, +\infinity)).
        l can be an array.
        '''
        return -self.C1 + np.exp(l - self.C2)

    def predict_failure(self, hi):
        '''
        Failure probability at next epoch.
        '''
        return _sigmoid(hi)



def fit_1ParamBrownian(hi_list: list[np.array], t: int = 1):
    '''
    Fit a 1-parameter Brownian motion model to the degradation data.
    '''
    n = len(hi_list)
    theta_list = [0.0] * n
    brownian_list = [None] * n
    for i, hi in enumerate(hi_list):
        resInc = np.diff(hi, prepend = 0)
        tmp = np.mean(resInc)
        theta_list[i] = tmp / t
        brownian_list[i] = resInc - tmp
    theta_list = np.array(theta_list)
    mu0, sigma0_square = np.mean(theta_list), np.var(theta_list, ddof=1)
    brownian_list = np.concatenate(brownian_list)
    sigma_square = np.var(brownian_list, ddof=1)
    return mu0, sigma0_square, sigma_square

def fit_2ParamBrownian(hi_list: list[np.array], t: int = 1):
    '''
    Fit a 2-parameter Brownian motion model to the degradation data.
    '''
    theta1_list = np.array([hi[0] for hi in hi_list])
    new_hi_list = [hi - hi[0] for hi in hi_list]
    mu0 = np.mean(theta1_list)
    sigma0_square = np.var(theta1_list, ddof=1)
    mu1, sigma1_square, sigma_square = fit_1ParamBrownian(new_hi_list, t)
    return mu0, sigma0_square, mu1, sigma1_square, sigma_square