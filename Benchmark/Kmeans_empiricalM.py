import numpy as np
from time import time
from Kmeans_utils import brownian_motion_sample, empirical_k_means_measure
from LP_utils import gurobi_bm

np.random.seed(12345)

def cost(x, y):
    return np.sum((x - y)**2)

T = 41 # including time 0
# IMPORTANT: Number of cubes is n^{1/T} by ESTIMATING PROCESSES IN ADAPTED WASSERSTEIN DISTANCE
n = 1000
x_vol = 1.0
y_vol = 0.5
x_init = 1.0
y_init = 2.0
N_INSTANCE = 1
val_results = np.zeros(N_INSTANCE)
t0 = time()

for ins in range(N_INSTANCE):
    bms = brownian_motion_sample(T-1, n, vol=x_vol) + x_init
    mu_full = [bms, np.ones(len(bms)) / len(bms)]
    # mu_ad_x, mu_ad_w = empirical_k_means_measure(bms, use_weights=1)
    partition_list = np.zeros(T + 1, dtype=int)  # [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    partition_list[0] = 1
    partition_list[1:11] = 2
    partition_list[11:] = 2
    mu_ad_x, mu_ad_w = empirical_k_means_measure(bms, use_klist=1, klist=partition_list, use_weights=1)
    # ad_x is path list, ad_w is probability
    mu_full_ad = [mu_ad_x, mu_ad_w]

    bms_nu = brownian_motion_sample(T-1, n, vol=y_vol) + y_init
    nu_full = [bms_nu, np.ones(len(bms_nu)) / len(bms_nu)]
    nu_ad_x, nu_ad_w = empirical_k_means_measure(bms_nu, use_klist=1, klist=partition_list, use_weights=1)
    nu_full_ad = [nu_ad_x, nu_ad_w]

    print('Scenes after aggregation', mu_ad_x.shape, nu_ad_x.shape)
    val0, _ = gurobi_bm([mu_full_ad, nu_full_ad], f=cost, r_opti=1, causal=1, anticausal=1, outputflag=1)

    print('Sample points:', n, 'AOT value (adapted empirical):', val0-(x_init-y_init)**2, 'Time:', time() - t0)

    val_results[ins] = val0-(x_init-y_init)**2


print('Final results', val_results)
print('Mean', val_results.mean())
print('Standard deviation', val_results.std())
print('Avergage time', (time() - t0)/N_INSTANCE)
