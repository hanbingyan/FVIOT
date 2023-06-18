import numpy as np
from measures import rand_tree_binom
from LP_utils import get_marginals_multi, Graph, gurobi_bm
from time import time

np.random.seed(12345)

N_INSTANCE = 1
T = 2 # number of non-trivial time-steps (time 0 starting at 0 is not included)
N_BRANCH = 2
x_vol = 1.0
y_vol = 0.5
x_init = 1.0
y_init = 2.0


def cost(x, y):
    return np.sum((x-y)**2)

final_result = np.zeros(N_INSTANCE)
g = Graph(T+1)  # indices of the graph are 0, 1, 2
# Markovian structure:
for t in range(T):
    g.addEdge(t, t+1)

t0 = time()

for n_ins in range(N_INSTANCE):
    print('Instance:', n_ins)
    # mu, supp_mu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)
    # nu, supp_nu = rand_tree_pichler(T, num_branch=tuple([N_BRANCH] * T), udrange=UDRANGE)

    mu, supp_mu = rand_tree_binom(T, init=x_init, vol=x_vol, N_leaf=N_BRANCH, in_size=200)
    nu, supp_nu = rand_tree_binom(T, init=y_init, vol=y_vol, N_leaf=N_BRANCH, in_size=200)

    mu_marg_full = get_marginals_multi(mu, supp_mu, list(range(T + 1)), g, [])
    nu_marg_full = get_marginals_multi(nu, supp_nu, list(range(T + 1)), g, [])

    LP_bc_v1, _ = gurobi_bm([mu_marg_full, nu_marg_full], f=cost, r_opti=1, causal=1, anticausal=1, outputflag=0)

    final_result[n_ins] = LP_bc_v1 - (x_init - y_init)**2
    print('Values for LP bicausal', final_result[n_ins])


t_bc_2 = time() - t0

print('All final value:', final_result)
print('Final mean:', final_result.mean())
print('Final std:', final_result.std())
print('Average time for LP bicausal', t_bc_2/N_INSTANCE)
