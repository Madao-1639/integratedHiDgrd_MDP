import numpy as np

class HIMappingMixIn_Linear:
    """
    A mix-in class providing linear transformation methods between Health Index (hi) and degradation signal (l).

    This class is designed to be inherited by MDP classes that require bidirectional mapping between
    health indices and degradation signals. It assumes the subclass will provide a `l_min` attribute, which
    represents the minimum value of the degradation signal (serving as the offset for linear transformation).
    """
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



class HIMappingMixIn_Log:
    """
    A mix-in class providing logarithmic transformation methods between Health Index (hi) and degradation signal (l).

    This class is designed to be inherited by MDP classes that require bidirectional mapping between
    health indices and degradation signals. It assumes the subclass will provide `C1`, `C2` attributes. `C1` is the shift parameter in the logarithmic transformation, and `C2` is the offset for the degradation signal.
    """
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



class PredictFailureMixIn_Sigmoid:
    """
    A mix-in class providing a method to predict failure probability using a sigmoid function, based on health index (hi).

    This class is designed to be inherited by MDP classes that require failure prediction functionality.
    """
    def predict_failure(self, hi):
        '''
        Failure probability at next epoch.
        '''
        hi = hi.clip(-100, 100) # Avoid overflow
        # Sigmoid function
        p = np.where(
            hi >= 0,
            1 / (1 + np.exp(-hi)),
            np.exp(hi) / (1 + np.exp(hi))
        )
        return p