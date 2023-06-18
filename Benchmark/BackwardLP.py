import numpy as np
from measures import rand_tree_binom
from LP_utils import Graph
from BackwardLP_utils import solve_dynamic
from time import time

np.random.seed(12345)

N_INSTANCE = 10
T = 3 # number of non-trivial time-steps (time 0 starting at 0 is not included)
N_BRANCH = 2
x_vol = 1.0
y_vol = 0.5
x_init = 1.0
y_init = 2.0


def cost(x, y):
    return (x[0]-y[0])**2

final_result = np.zeros(N_INSTANCE)
g = Graph(T+1)  # indices of the graph are 0, 1, 2
# Markovian structure:
for t in range(T):
    g.addEdge(t, t+1)

t0 = time()

for n_ins in range(N_INSTANCE):
    print('Instance:', n_ins)
    mu, supp_mu = rand_tree_binom(T, init=x_init, vol=x_vol, N_leaf=N_BRANCH, in_size=200) # insize=200
    nu, supp_nu = rand_tree_binom(T, init=y_init, vol=y_vol, N_leaf=N_BRANCH, in_size=200)

    # Backward induction
    cost_funs = [[[t], cost] for t in range(T+1)]
    t1 = time()
    BW_v1, _ = solve_dynamic(cost_funs, mu, nu, supp_mu, supp_nu, g, outputflag=0, method='gurobi')
    BW_v1 = BW_v1[0] - (x_init - y_init)**2

    print('Values for Backward induction bicausal', BW_v1)
    print('sampling time', t1-t0)
    print('algo time', time()-t1)
    final_result[n_ins] = BW_v1

t_bc_2 = time() - t0

print('All final value:', final_result)
print('Final mean:', final_result.mean())
print('Final std:', final_result.std())
print('Average time for LP bicausal', t_bc_2/N_INSTANCE)

