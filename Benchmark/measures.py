import numpy as np
from time import time
from scipy.stats import norm
from sklearn.cluster import KMeans




# FUNCTIONS TO RESHAPE MEASURES FOR CAUSAL AND BICAUSAL VERSION OF SINKHORN'S ALGORITHM
def get_meas_for_sinkhorn(mu, supp_mu, t_max, markov=1):
    """
    Only implemented for Markovian measures at the moment
    mu and mu_supp are as usual for DPP
    Returns:
    x_list gives the supports at each time step
    a simply measure will be represented by a list of indices, and an array of weights, corresponding to the indices
    so mu_list[0] is just such a simply measure
    On the other hand, mu_list[i] for i > 0 will be a list, each entry k corresponding to the mu_i(x_k, \cdot),
    so mu_list[i] is basically the kernel mu_i(\cdot, \cdot), and each entry k specifies the value (i.e., the measure)
    """

    x_list = [supp_mu([i]).flatten() for i in range(t_max)]
    mu_list = []
    mu0_s_pre = mu(0, [])[0].flatten()
    w0_pre = np.array(mu(0, [])[1])
    mu0 = []
    w0 = []
    for ind in range(len(mu0_s_pre)):
        for ind2, x_h in enumerate(x_list[0]):
            if mu0_s_pre[ind] == x_h:
                if ind2 in mu0:
                    i_where = mu0.index(ind2)
                    w0[i_where] += w0_pre[ind]
                else:
                    mu0.append(ind2)
                    w0.append(w0_pre[ind])
                break
    mu_list.append([tuple(mu0), np.array(w0)])

    for t in range(1, t_max):
        mu_h = []
        for ind_minus, x_minus in enumerate(x_list[t-1]):
            mut_s_pre = mu(t, [x_minus])[0].flatten()
            wt_pre = np.array(mu(t, [x_minus])[1])
            mut = []
            wt = []
            for ind in range(len(mut_s_pre)):
                for ind2, x_h in enumerate(x_list[t]):
                    if mut_s_pre[ind] == x_h:
                        if ind2 in mut:
                            i_where = mut.index(ind2)
                            wt[i_where] += wt_pre[ind]
                        else:
                            mut.append(ind2)
                            wt.append(wt_pre[ind])
            mu_h.append([tuple(mut), np.array(wt)])
        mu_list.append(mu_h)

    return x_list, mu_list



def get_full_index_markov(mu_list):
    """
    should return a list of the same length as mu_list, where each entry represents the support of mu_{1:t} in the
    "full-index"-version. I.e., each entry of out[t], i.e., out[t][k], is a list where the j=0 entry is ind_0,
    the j=1 entry is ind_{0:0}, the j=2 entry is ind_1, the j=3 entry is ind_{0:1}, ...,
    the 2j entry is ind_{j} and the 2j+1 entry is ind_{0:j} (for j up to t). k ranges through the number of different
    support points of mu_{1:t}
    """
    if len(mu_list) == 1:
        return [[[mu_list[0][0][i], mu_list[0][0][i]] for i in range(len(mu_list[0][0]))]]
    else:
        glprev_full = get_full_index_markov(mu_list[:-1])
        glprev = glprev_full[-1]
        out_l = []
        itot_1t = 0
        for i_tot_prev in glprev:
            i_sup_next = i_tot_prev[-2]
            sh_list = mu_list[-1][i_sup_next][0]
            for sh in sh_list:
                out_l_add = i_tot_prev.copy()
                out_l_add.append(sh)
                out_l_add.append(itot_1t)
                itot_1t += 1
                out_l.append(out_l_add.copy())
        gl_out = glprev_full.copy()
        gl_out.append(out_l)
        return gl_out


def get_start_next_indices(gl_out):
    # for causal version of Sinkhorn's algorithm.
    # output is a list with t_max-1 entries, each being a list of numbers, say out_t
    # out_t[i]:out_t[i+1] gives the range of indices for the joint support of mu_{1:t+1} which share the same elements
    # x_{1:t} (so in this range of indices, only x_{t+1} varies).
    t_max = len(gl_out)
    out_next_l = []
    for t in range(1, t_max):
        out_h = [0]
        i_prev_cur = 0
        i_h = gl_out[t]
        for i_cur in i_h:
            if i_cur[-3] > i_prev_cur:
                out_h.append(i_cur[-1])
                i_prev_cur = i_cur[-3]
        out_h.append(i_h[-1][-1]+1)
        out_next_l.append(out_h)
    return out_next_l


def get_joint_prob(nu_list, nu_index_full, t_ind):
    ih = nu_index_full[t_ind]
    out_prob = []
    for i_full in ih:
        p_here = nu_list[0][1][i_full[0]]
        for t in range(1, t_ind+1):
            ind = nu_list[t][i_full[2 * (t - 1)]][0].index(i_full[2*t])
            p_here *= nu_list[t][i_full[2*(t-1)]][1][ind]
        out_prob.append(p_here)
    return out_prob


def sinkhorn_bicausal_markov(mu_list, nu_list, cost_list, n_list, m_list, eps_stop=10**-4, max_iter=10**4,
                             reshape=True, outputflag=0):
    # Only for MARKOV - MARKOV marginals, bicausal!
    """

    :param mu_list: as output by get_meas_for_sinkhorn
    :param nu_list: as output by get_meas_for_sinkhorn
    :param cost_list: list of matrices, one for each time point (markov case). Notably, the cost functions should
                    already be kernelized, i.e., values are exp(-c) instead of c
    :param n_list: sizes of supports for mu for each time step
    :param m_list: sizes of supports for nu for each time step
    :return:
    """
    t_max = len(mu_list)

    # initializing dual functions. We specify them in a multiplicative way, i.e., compared to the paper, we store values
    # of exp(f_t) and exp(g_t) instead of f_t and g_t, which is in line with standard implementations of Sinkhorn's
    tinit = time()
    f_1 = np.ones(n_list[0])
    g_1 = np.ones(m_list[0])
    f_list = [f_1]
    g_list = [g_1]
    const_f_list = [0]
    const_g_list = [0]
    for t in range(1, t_max):
        f_h = [[np.ones([len(mu_list[t][i][1]), 1]) for j in range(m_list[t-1])] for i in range(n_list[t-1])]
        g_h = [[np.ones([1, len(nu_list[t][j][1])]) for j in range(m_list[t-1])] for i in range(n_list[t-1])]
        c_f_h = [[1 for j in range(m_list[t-1])] for i in range(n_list[t-1])]
        c_g_h = [[1 for j in range(m_list[t-1])] for i in range(n_list[t-1])]
        f_list.append(f_h)
        g_list.append(g_h)
        const_f_list.append(c_f_h)
        const_g_list.append(c_g_h)
    if outputflag:
        print('Initializing took ' + str(time()-tinit) + ' seconds')

    # Define update iterations:
    t_funs = time()
    def update_f_t(mut, nut, gt, ct):
        """

        :param mut: should be of shape (a, 1)
        :param nut: should be of shape (1, b)
        :param gt: should be of shape (1, b)
        :param ct: should be of shape (a, b)
        :return: array of shape (a, 1) representing f_t
        """
        # at = 1. / np.sum(gt * ct * nut, axis=1, keepdims=True)
        # at = 1. / np.dot(ct, (gt*nut).T)
        at = 1. / np.matmul(ct, (gt*nut).T)
        cth = np.sum(np.log(at) * mut)
        return at/np.exp(cth), cth

    def update_g_t(mut, nut, ft, ct):
        """

        :param mut: should be of shape (a, 1)
        :param nut: should be of shape (1, b)
        :param ft: should be of shape (a, 1)
        :param ct: should be of shape (a, b)
        :return: array of shape (1, b) representing g_t
        """
        # bt = 1. / np.sum(ft*ct*mut, axis=0, keepdims=True)
        # bt = 1. / np.dot(ct.T, ft*mut).T
        bt = 1. / np.matmul(ct.T, ft*mut).T
        cth = np.sum(np.log(bt) * nut)
        return bt/np.exp(cth), cth

    def update_f_1(mut, nut, gt, ct):
        # inputs as for update_f_t
        at = 1. / np.sum(gt * ct * nut, axis=1, keepdims=True)
        return at, np.sum(np.log(at) * mut)

    def update_g_1(mut, nut, ft, ct):
        # inputs as for update_g_t
        bt = 1. / np.sum(ft * ct * mut, axis=0, keepdims=True)
        return bt, np.sum(np.log(bt) * nut)

    def full_update_f_list():
        for t_m in range(t_max):
            t = t_max - t_m - 1
            if t > 0:
                cvnew = np.ones([n_list[t-1], m_list[t-1]])
            if t == 0:
                f_list[0], value_f = update_f_1(mu_list[0][1], nu_list[0][1], g_list[0], cost_list[0][mu_list[0][0], :][:, nu_list[0][0]] * cvh[mu_list[0][0], :][:, nu_list[0][0]])
            elif t == t_max-1:
                for i in range(n_list[t-1]):
                    for j in range(m_list[t-1]):
                        f_list[t][i][j], cvnew[i, j] = update_f_t(mu_list[t][i][1], nu_list[t][j][1], g_list[t][i][j], cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]])
            else:
                for i in range(n_list[t-1]):
                    for j in range(m_list[t-1]):
                        f_list[t][i][j], cvnew[i, j] = update_f_t(mu_list[t][i][1], nu_list[t][j][1], g_list[t][i][j], cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]]
                                                                  * cvh[mu_list[t][i][0], :][:, nu_list[t][j][0]])
            cvh = np.exp(-cvnew.copy())
            const_f_list[t] = cvh.copy()
        return value_f

    def full_update_g_list():
        for t_m in range(t_max):
            t = t_max - t_m - 1
            if t > 0:
                cvnew = np.ones([n_list[t-1], m_list[t-1]])
            if t == 0:
                g_list[0], value_g = update_g_1(mu_list[0][1], nu_list[0][1], f_list[0], cost_list[0][mu_list[0][0], :][:, nu_list[0][0]] * cvh[mu_list[0][0], :][:, nu_list[0][0]])
            elif t == t_max-1:
                for i in range(n_list[t-1]):
                    for j in range(m_list[t-1]):
                        g_list[t][i][j], cvnew[i, j] = update_g_t(mu_list[t][i][1], nu_list[t][j][1], f_list[t][i][j], cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]])
            else:
                for i in range(n_list[t-1]):
                    for j in range(m_list[t-1]):
                        g_list[t][i][j], cvnew[i, j] = update_g_t(mu_list[t][i][1], nu_list[t][j][1], f_list[t][i][j], cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]]
                                                                  * cvh[mu_list[t][i][0], :][:, nu_list[t][j][0]])
            cvh = np.exp(-cvnew.copy())
            const_g_list[t] = cvh.copy()
        return value_g

    if outputflag:
        print('Defining update functions took ' + str(time()-t_funs) + ' seconds')

    if reshape:
        # reshape inputs
        # we want mu_list[t][i][1] to be shaped (a, 1) and nu_list[t][j][1] to be shaped (1, b) for some a and b that may
        # depend on i and j
        t_reshape = time()
        for t in range(t_max):
            if t == 0:
                if len(mu_list[t][1].shape) == 1:
                    mu_list[t][1] = np.expand_dims(mu_list[t][1], 1)
                if len(nu_list[t][1].shape) == 1:
                    nu_list[t][1] = np.expand_dims(nu_list[t][1], 0)
                if len(mu_list[t]) == 2:
                    mu_list[t].append(np.log(mu_list[t][1]))
                if len(nu_list[t]) == 2:
                    nu_list[t].append(np.log(nu_list[t][1]))
            else:
                for i in range(n_list[t-1]):
                    if len(mu_list[t][i][1].shape) == 1:
                        mu_list[t][i][1] = np.expand_dims(mu_list[t][i][1], 1)
                    if len(mu_list[t][i]) == 2:
                        mu_list[t][i].append(np.log(mu_list[t][i][1]))

                for j in range(m_list[t-1]):
                    if len(nu_list[t][j][1].shape) == 1:
                        nu_list[t][j][1] = np.expand_dims(nu_list[t][j][1], 0)
                    if len(nu_list[t][j]) == 2:
                        nu_list[t][j].append(np.log(nu_list[t][j][1]))

        if outputflag:
            print('Reshaping input took ' + str(time()-t_reshape) + ' seconds')

    t_solve = time()
    prev_val = -10**8
    value_f = -100
    value_g = -100
    iter_h = 0
    while iter_h < max_iter and np.abs(prev_val - value_f - value_g) > eps_stop:
        if iter_h % 10 == 0 and outputflag:
            print('Current iteration:', iter_h, 'Current value:', value_f+value_g, 'Current time:', time()-t_solve)
        iter_h += 1
        prev_val = value_f + value_g
        value_f = full_update_f_list()
        value_g = full_update_g_list()
        # print(value_f)
        # print(value_g)
    if outputflag:
        print('Solving took ' + str(time()-t_solve) + ' seconds')

    # get value without entropy
    for t_m in range(t_max):
        t = t_max - t_m - 1
        if t > 0:
            V_t = np.zeros([n_list[t-1], m_list[t-1]])
        if t == t_max-1:
            for i in range(n_list[t-1]):
                for j in range(m_list[t-1]):
                    V_t[i, j] = np.sum(-np.log(cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]]) * f_list[t][i][j] * g_list[t][i][j] * cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]] * (1./const_g_list[t][i, j]) * mu_list[t][i][1] * nu_list[t][j][1])
        elif t > 0:
            for i in range(n_list[t-1]):
                for j in range(m_list[t-1]):
                    V_t[i, j] = np.sum((-np.log(cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]]) + V_tp[mu_list[t][i][0], :][:, nu_list[t][j][0]])
                                       * f_list[t][i][j] * g_list[t][i][j] * cost_list[t][mu_list[t][i][0], :][:, nu_list[t][j][0]] *
                                       const_g_list[t+1][mu_list[t][i][0], :][:, nu_list[t][j][0]] * (1./const_g_list[t][i, j]) * mu_list[t][i][1] * nu_list[t][j][1])
        else:
            value = np.sum((-np.log(cost_list[0][mu_list[0][0], :][:, nu_list[0][0]]) + V_tp[mu_list[t][0], :][:, nu_list[t][0]]) * f_list[0] * g_list[0] * cost_list[0][mu_list[0][0], :][:, nu_list[0][0]] * const_g_list[t+1][mu_list[t][0], :][:, nu_list[t][0]] * mu_list[0][1] * nu_list[0][1])
        V_tp = V_t.copy()
    return value
    # return 0.0

def rand_tree_binom(T, init, vol, N_leaf=2, in_size=100):
    # Trying to implement the method in Chapter 5 from https://arxiv.org/pdf/2102.05413.pdf

    transitions = {}  # take as key a tuple of an (integer node and integer value) and returns a measure
    supports = {}  # takes as input a node and returns a set of integer support points
    for i in range(T+1):
        supports[i] = set([])

    for t in range(T):
        if t == 0:
            supports[0] = {init}
            rand_supps = np.random.randn(N_leaf*in_size, 1)*vol
            kmeans = KMeans(n_clusters=N_leaf).fit(rand_supps)
            _, probs = np.unique(kmeans.labels_, return_counts=True)
            # probs = np.ones(N_leaf)
            probs = probs/np.sum(probs)
            supps = kmeans.cluster_centers_
            # supps = np.random.random(size=[N_leaf, 1])
            # supps = supps.reshape((-1, 1))
            supps_int = set(np.squeeze(init+supps, axis=1))
            # supps_int = set(supps)
            supports[1] |= supps_int
            transitions[(t + 1, init)] = [init+supps, probs]
        else:
            for x_int in supports[t]:
                rand_supps = np.random.randn(N_leaf*in_size, 1)*vol
                kmeans = KMeans(n_clusters=N_leaf).fit(rand_supps)
                _, probs = np.unique(kmeans.labels_, return_counts=True)
                # probs = np.ones(N_leaf)
                probs = probs / np.sum(probs)
                supps = kmeans.cluster_centers_
                # probs = np.ones(N_leaf)
                # probs = probs/np.sum(probs)
                # supps = np.random.random(size=[N_leaf, 1])
                # supps = supps.reshape((-1, 1))
                supps_int = set(np.squeeze(x_int+supps, axis=1))
                # supps_int = set(x_int + supps)
                supports[t + 1] |= supps_int
                transitions[(t + 1, x_int)] = [x_int+supps, probs]

    def mu(node, x_parents):
        if node == 0:
            return [np.reshape(np.array([init]), (-1, 1)), [1]]
        x = x_parents[0]  # should only contain one element as the structure is Markovian
        # x = int(x)
        return transitions[(node, x)]

    def sup_mu(node_list):
        if len(node_list) == 0:
            out = np.array([])
            out = out.reshape(-1, 1)
            return out
        return np.reshape(np.array(list(supports[node_list[0]])), (-1, 1))  # we only supply support for single nodes

    print('Warning: You are using a measure where only one-step supports are specified')
    return mu, sup_mu




# def tree_approx(T, init, vol, n_grid=50):
#     # Trying to implement the method in Chapter 5 from https://arxiv.org/pdf/2102.05413.pdf
#
#     transitions = {}  # take as key a tuple of an (integer node and integer value) and returns a measure
#     supports = {}  # takes as input a node and returns a set of integer support points
#     candidate_arr = []
#     for i in range(T+1):
#         candidate = np.arange(i*n_grid+1)
#         candidate = candidate - candidate.mean() + init
#         candidate = candidate.astype(int)
#         candidate_arr.append(candidate)
#         supports[i] = set([])
#
#     for t in range(T):
#         if t == 0:
#             supports[0] = {init}
#             probs = np.zeros_like(candidate_arr[t+1], dtype=float)
#             for k in range(len(probs)):
#                 probs[k] = norm.cdf(candidate_arr[t+1][k]+0.5, loc=init, scale=vol) - \
#                            norm.cdf(candidate_arr[t+1][k]-0.5, loc=init, scale=vol)
#             nonzero_prob = probs > 1e-5
#             supps = candidate_arr[t+1][nonzero_prob]
#             supps_int = set(supps)
#             probs = probs[nonzero_prob]
#             probs = probs/probs.sum()
#             supports[1] |= supps_int
#             transitions[(t + 1, init)] = [supps, probs]
#         else:
#             for x_int in supports[t]:
#                 probs = np.zeros_like(candidate_arr[t+1], dtype=float)
#                 for k in range(len(probs)):
#                     probs[k] = norm.cdf(candidate_arr[t+1][k] + 0.5, loc=x_int, scale=vol) - \
#                                norm.cdf(candidate_arr[t+1][k] - 0.5, loc=x_int, scale=vol)
#                 nonzero_prob = probs > 1e-5
#                 supps = candidate_arr[t+1][nonzero_prob]
#                 supps_int = set(supps)
#                 probs = probs[nonzero_prob]
#                 probs = probs/probs.sum()
#                 supports[t+1] |= supps_int
#                 transitions[(t+1, x_int)] = [supps, probs]
#
#     def mu(node, x_parents):
#         if node == 0:
#             return [np.reshape(np.array([init]), (-1, 1)), [1]]
#         x = x_parents[0]  # should only contain one element as the structure is Markovian
#         # x = int(x)
#         return transitions[(node, x)]
#
#     def sup_mu(node_list):
#         if len(node_list) == 0:
#             out = np.array([])
#             out = out.reshape(-1, 1)
#             return out
#         return np.reshape(np.array(list(supports[node_list[0]])), (-1, 1))  # we only supply support for single nodes
#
#     print('Warning: You are using a measure where only one-step supports are specified')
#     return mu, sup_mu
#
#
# def adapted_tree(T, init, vol, grid_size=0.1, n_grid=51, in_size=2):
#     # Trying to implement the method in Chapter 5 from https://arxiv.org/pdf/2102.05413.pdf
#
#     transitions = {}  # take as key a tuple of an (integer node and integer value) and returns a measure
#     supports = {}  # takes as input a node and returns a set of integer support points
#     candidate_arr = []
#     for i in range(T+1):
#         candidate = np.arange(i*n_grid+1)*grid_size
#         candidate = candidate - candidate.mean() + init
#         candidate_arr.append(candidate)
#         supports[i] = set([])
#
#     # sample data
#     data = np.zeros((1000, T+1))
#     data[:, 0] = init
#     for idx in range(1000):
#         for t in range(T):
#             data[idx, t+1] = data[idx, t] + vol*np.random.randn(1)
#
#     for t in range(T):
#         if t == 0:
#             supports[0] = {init}
#             rand_supps = data[:, t+1]
#             labels = np.zeros_like(rand_supps, dtype=np.int)
#             for k in range(len(rand_supps)):
#                 labels[k] = np.argmin(np.abs(candidate_arr[t+1] - rand_supps[k]))
#             uni_labels, probs = np.unique(labels, return_counts=True)
#             probs = probs/probs.sum()
#             supps = candidate_arr[t+1][uni_labels]
#             supps_int = set(supps)
#             supports[1] |= supps_int
#             transitions[(t + 1, init)] = [supps, probs]
#         else:
#             for x_int in supports[t]:
#                 # find where data == x_int at time t
#                 cond_idx = []
#                 for k in range(1000):
#                     min_val = candidate_arr[t][np.argmin(np.abs(data[k, t] - candidate_arr[t]))]
#                     if np.abs(min_val - x_int) < grid_size/10:
#                         cond_idx.append(k)
#                 cond_idx = np.array(cond_idx, dtype=int)
#                 rand_supps = data[cond_idx, t+1]
#                 labels = np.zeros_like(rand_supps, dtype=np.int)
#                 for k in range(len(rand_supps)):
#                     labels[k] = np.argmin(np.abs(candidate_arr[t+1] - rand_supps[k]))
#                 uni_labels, probs = np.unique(labels, return_counts=True)
#                 probs = probs/probs.sum()
#                 supps = candidate_arr[t+1][uni_labels]
#                 supps_int = set(supps)
#                 supports[t+1] |= supps_int
#                 transitions[(t+1, x_int)] = [supps, probs]
#
#     def mu(node, x_parents):
#         if node == 0:
#             return [np.reshape(np.array([init]), (-1, 1)), [1]]
#         x = x_parents[0]  # should only contain one element as the structure is Markovian
#         # x = int(x)
#         return transitions[(node, x)]
#
#     def sup_mu(node_list):
#         if len(node_list) == 0:
#             out = np.array([])
#             out = out.reshape(-1, 1)
#             return out
#         return np.reshape(np.array(list(supports[node_list[0]])), (-1, 1))  # we only supply support for single nodes
#
#     print('Warning: You are using a measure where only one-step supports are specified')
#     return mu, sup_mu
#
#
# def rand_tree_pichler(T, num_branch=(2, 3, 2, 3, 4), init=10, udrange=10, discr=0):
#     # Trying to implement the method in Chapter 5 from https://arxiv.org/pdf/2102.05413.pdf
#
#     transitions = {}  # take as key a tuple of an (integer node and integer value) and returns a measure
#     supports = {}  # takes as input a node and returns a set of integer support points
#     for i in range(T+1):
#         supports[i] = set([])
#
#     for t in range(T):
#         if t == 0:
#             if discr==0:
#                 supports[0] = {init}
#             else:
#                 supports[0] = {0}
#             nbh = num_branch[0]
#             probs = np.random.random_sample(nbh)
#             probs = probs/np.sum(probs)
#             if not discr:
#                 supps = np.random.randint(-udrange, udrange, size=[nbh, 1])
#                 supps_int = set(np.squeeze(10+supps, axis=1))
#                 supports[1] |= supps_int
#                 transitions[(t + 1, init)] = [10 + supps, probs]
#             else:
#                 supps = np.arange(0, nbh)
#                 supps = supps.reshape(-1, 1)
#                 supps_int = set(np.squeeze(supps, axis=1))
#                 supports[1] |= supps_int
#                 transitions[(t + 1, 0)] = [supps, probs]
#         else:
#             for x_int in supports[t]:
#                 nbh = num_branch[t]
#                 probs = np.random.random_sample(nbh)
#                 probs = probs/np.sum(probs)
#                 if not discr:
#                     supps = np.random.randint(-udrange, udrange, size=[nbh, 1])
#                     supps_int = set(np.squeeze(x_int+supps, axis=1))
#                     supports[t + 1] |= supps_int
#                     transitions[(t + 1, x_int)] = [x_int + supps, probs]
#                 else:
#                     supps = np.arange(0, nbh)
#                     supps = supps.reshape(-1, 1)
#                     supps_int = set(np.squeeze(supps, axis=1))
#                     supports[t+1] |= supps_int
#                     transitions[(t+1, x_int)] = [supps, probs]
#
#     if discr == 0:
#         def mu(node, x_parents):
#             if node == 0:
#                 return [np.reshape(np.array([10]), (-1, 1)), [1]]
#             x = x_parents[0]  # should only contain one element as the structure is Markovian
#             x = int(x)
#             return transitions[(node, x)]
#     else:
#         def mu(node, x_parents):
#             if node == 0:
#                 return [np.reshape(np.array([0]), (-1, 1)), [1]]
#             x = x_parents[0]  # should only contain one element as the structure is Markovian
#             x = int(x)
#             return transitions[(node, x)]
#
#     def sup_mu(node_list):
#         if len(node_list) == 0:
#             out = np.array([])
#             out = out.reshape(-1, 1)
#             return out
#         return np.reshape(np.array(list(supports[node_list[0]])), (-1, 1))  # we only supply support for single nodes
#
#     print('Warning: You are using a measure where only one-step supports are specified')
#     return mu, sup_mu