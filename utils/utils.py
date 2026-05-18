import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    '''Set random seed for reproducibility across all libraries.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def test4norm(hi_dict: dict, sig_list: tuple = (0.01, 0.05, 0.10), prepend: int = 0) -> dict:
    '''Test health indicator increments for normality across UUTs.

    Args:
        hi_dict (dict): Mapping of UUT identifiers to health indicator arrays.
        sig_list (tuple): Significance levels to test. Defaults to (0.01, 0.05, 0.10).
        prepend (int): Value to prepend before differencing. Defaults to 0.

    Returns:
        dict: Summary of normality test pass counts per test and significance level.
    '''
    from scipy.stats import kstest, shapiro, normaltest, anderson
    nt_summary = {}
    for UUT, hi in hi_dict.items():
        resInc = np.diff(hi, prepend=prepend)
        nt_result = {
            'KS': kstest(resInc, cdf='norm'),
            'SW': shapiro(resInc),
            'DP': normaltest(resInc),
            'AD': anderson(resInc, dist='norm')
        }
        for test_name, test_result in nt_result.items():
            if test_name == 'AD':
                for sig in sig_list:
                    ad_idx = np.where(test_result.significance_level == sig * 100)[0][0]
                    critical_value = test_result.critical_values[ad_idx]
                    statistic = test_result.statistic
                    if statistic < critical_value:
                        nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})', 0) + 1
                    else:
                        break
            else:
                statistic, p_value = test_result
                for sig in sig_list:
                    if p_value > sig:
                        nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})', 0) + 1
                    else:
                        break
    return nt_summary

def plot_hi(hi_dict: dict, record_UUTs: list | None = None, **fig_kwargs):
    '''Plot health indicator trajectories for specified UUTs.

    Args:
        hi_dict (dict): Mapping of UUT identifiers to health indicator arrays.
        record_UUTs (list, optional): List of UUT identifiers to plot. If None, all UUTs in hi_dict will be plotted. Defaults to None.
        **fig_kwargs: Keyword arguments passed to plt.figure().

    Returns:
        matplotlib.figure.Figure: The generated figure.
    '''
    import matplotlib.pyplot as plt
    # Default figure parameters
    fig_kwargs['figsize'] = fig_kwargs.get('figsize', (10, 5))

    fig = plt.figure(**fig_kwargs)
    record_UUTs = record_UUTs or hi_dict.keys()
    for UUT in record_UUTs:
        hi = hi_dict[UUT]
        plt.plot(hi, '-', lw=0.5, alpha=0.75, label=UUT)
    plt.legend()
    plt.tight_layout()
    return fig