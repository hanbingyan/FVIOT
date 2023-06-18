import numpy as np
from measures import get_meas_for_sinkhorn, get_full_index_markov, get_start_next_indices, \
    get_joint_prob, sinkhorn_bicausal_markov, rand_tree_binom #, rand_tree_pichler, tree_approx, adapted_tree
from time import time

np.random.seed(12345)
# Program produces values for Table 2 in the paper Eckstein & Pammer "Computational methods ..."
# Values for the table can be obtained directly from the console output

N_INSTANCE = 10
T = 2 # number of non-trivial time-steps (time 0 starting at 0 is not included)
N_BRANCH = 2
EPS = 0.1
x_vol = 1.0
y_vol = 0.5
x_init = 1.0
y_init = 2.0

def cost_f_scalar_2(x, y):
    return np.abs(x-y)**2

final_result = np.zeros(N_INSTANCE)

t0 = time()

for n_ins in range(N_INSTANCE):
    print('Instance:', n_ins)

    # mu, supp_mu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)
    # nu, supp_nu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)

    mu, supp_mu = rand_tree_binom(T, init=x_init, vol=x_vol, N_leaf=N_BRANCH, in_size=200)
    nu, supp_nu = rand_tree_binom(T, init=y_init, vol=y_vol, N_leaf=N_BRANCH, in_size=200)

    # mu, supp_mu = tree_approx(T, init=x_init, vol=x_vol, n_grid=10)
    # nu, supp_nu = tree_approx(T, init=y_init, vol=y_vol, n_grid=10)

    # mu, supp_mu = adapted_tree(T, init=x_init, vol=x_vol, grid_size=0.1, n_grid=41, in_size=30)
    # nu, supp_nu = adapted_tree(T, init=y_init, vol=y_vol, grid_size=0.1, n_grid=41, in_size=30)

    # Sinkhorn
    x_list, mu_list = get_meas_for_sinkhorn(mu, supp_mu, T + 1)
    y_list, nu_list = get_meas_for_sinkhorn(nu, supp_nu, T + 1)
    ind_tot = get_full_index_markov(nu_list)
    ind_next_l = get_start_next_indices(ind_tot)
    nu_joint_prob = get_joint_prob(nu_list, ind_tot, T - 1)
    cost_mats_2 = []
    for t in range(T + 1):
        cmh_2 = np.zeros([len(x_list[t]), len(y_list[t])], dtype=np.float64)
        # if t == T:
        for i in range(len(x_list[t])):
            for j in range(len(y_list[t])):
                cmh_2[i, j] = np.exp(-1 / EPS * cost_f_scalar_2(x_list[t][i], y_list[t][j]))
        cost_mats_2.append(cmh_2)

    n_list = [len(x_list[i]) for i in range(T + 1)]
    m_list = [len(y_list[i]) for i in range(T + 1)]

    print('n_list', n_list)
    print('m_list', m_list)


    val_sink_2 = sinkhorn_bicausal_markov(mu_list, nu_list, cost_mats_2, n_list, m_list, eps_stop=10**-4, max_iter=1000,
                                          reshape=True, outputflag=0)


    sink_bc_v2 = val_sink_2 * EPS

    final_result[n_ins] = sink_bc_v2 - (x_init - y_init)**2
    print('Values for Sinkhorn (bicausal), EPS = ' + str(EPS), final_result[n_ins])


t_sink_bc_2 = time()-t0

print('All final value:', final_result)
print('Final mean:', final_result.mean())
print('Final std:', final_result.std())
print('Average time for Sinkhorn (bicausal), EPS = ' + str(EPS), t_sink_bc_2/N_INSTANCE)
