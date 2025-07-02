import os
import random
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def test4norm(hi_dict,sig_list = (0.01,0.05,0.10)):
    '''Test for normality'''
    from scipy.stats import kstest,shapiro,normaltest,anderson
    nt_summary = {}
    for UUT,hi in hi_dict.items():
        resInc = np.diff(hi,prepend=0)
        nt_result = {
            'KS': kstest(resInc,cdf='norm'),
            'SW': shapiro(resInc),
            'DP': normaltest(resInc),
            'AD': anderson(resInc,dist='norm')
        }
        for test_name, test_result in nt_result.items():
            if test_name == 'AD':
                for sig in sig_list:
                    ad_idx = np.where(test_result.significance_level==sig*100)[0][0]
                    critical_value = test_result.critical_values[ad_idx]
                    statistic = test_result.statistic
                    if statistic < critical_value:
                        nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                    else:
                        break
            else:
                statistic, p_value = test_result
                for sig in sig_list:
                    if p_value > sig:
                        nt_summary[f'{test_name} pass({sig:.0%})'] = nt_summary.get(f'{test_name} pass({sig:.0%})',0) + 1
                    else:
                        break
    return nt_summary

def plot_hi(hi_dict,record_UUTs,**fig_kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure(**fig_kwargs)
    for UUT in record_UUTs:
        hi = hi_dict[UUT]
        plt.plot(hi,'-',lw=0.5,alpha=0.75,label=UUT)
    plt.legend()
    plt.tight_layout()
    return fig