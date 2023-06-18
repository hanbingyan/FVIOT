import numpy as np
from collections import defaultdict
from itertools import product
from gurobipy import *

##################### LP helper functions #################
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices
        self.parents = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.parents[v].append(u)

        # A recursive function used by topologicalSort

    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Push current vertex to stack which stores result
        stack.insert(0, v)

        # The function to do Topological Sort. It uses recursive

    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Print contents of stack
        return stack


# FUNCTION TO CHANGE REPRESENTATION OF MEASURES
def reduce_meas(marg, filt=0):
    # marg is a list with two entries, one n x d np array, and one n-list of weights
    # the goal of this function is to identify duplicates in the first entry and reduce the representation
    if filt == 1:
        return marg  # TODO: check how to reduce in this case as well ...
    if len(marg[0].shape) == 1:
        marg[0].reshape(-1, 1)
    uniques, inv = np.unique(marg[0], axis=0, return_inverse=1)
    w_new = np.zeros(len(uniques))
    for l in range(len(marg[0])):
        w_new[inv[l]] += marg[1][l]
    # for i in range(len(uniques)):
    #     for l in range(len(marg[0])):
    #         if np.all(uniques[i, :] == marg[0][l, :]):
    #             w_new[i] += marg[1][l]
    return [uniques, w_new]


def get_marginals_multi(mu, supp_mu, node_set, g, given_margs, all_out=0, index_mu=0, tol=10**-6, filt=0):
    # function should get a joint distribution on the specified node set. I.e., we want to go from disintegration
    # representation of a measure towards specifying the joint distribution.
    # node_set is a tuple containing the nodes that we wish to calculate the joint marginal on

    # get relevant parents
    rel_par = []
    for nh in node_set:
        rel_par.extend(g.parents[nh].copy())
    rel_par = list(tuple(rel_par))
    rel_par.sort()
    rel_par = tuple(rel_par)
    rel_par_arr = np.array(rel_par)

    # for each node in node_set, get the indices of the respective parents in rel_par
    # rel_par_arr[indices[j]] will give the parents of node_set[j]
    indices = []
    for nh in node_set:
        ph = g.parents[nh].copy()
        index = np.zeros(len(ph), dtype=int)
        for ind0, i in enumerate(ph):
            for ind in range(len(rel_par)):
                if rel_par[ind] == i:
                    index[ind0] = ind
        indices.append(index)

    # get relevant marginal rel_marg of the form [ n x d array, list of weights]
    if len(rel_par) == 0:
        # for each node in node_set, get the parent values from xh
        x_lists = []
        w_lists = []
        if filt == 1:
            filt_list_a = []
            filt_list_b = []
        for ind, nh in enumerate(node_set):
            if index_mu == 0:
                marg_x_nh = mu(nh, [])
            else:
                marg_x_nh = mu(nh, 0)
            if len(np.array(marg_x_nh[0]).shape) == 2:
                x_lists.append(marg_x_nh[0][:, 0])  # marg_x_nh[0] is always of shape [n, 1]
                if filt == 1:
                    filt_list_a.append(marg_x_nh[2][:, 0])
                    filt_list_b.append(marg_x_nh[2][:, 1])
            else:
                x_lists.append(marg_x_nh[0])
                if filt == 1:
                    filt_list_a.append(marg_x_nh[2][:, 0])
                    filt_list_b.append(marg_x_nh[2][:, 1])

            w_lists.append(marg_x_nh[1])

        x_list_comb = list(product(*x_lists))
        if filt == 1:
            filt_a_comb = list(product(*filt_list_a))
            filt_b_comb = list(product(*filt_list_b))
            a_arr = np.array(filt_a_comb)
            b_arr = np.array(filt_b_comb)
            if len(a_arr.shape) == 1:
                a_arr = a_arr.reshape(-1, 1)
            if len(b_arr.shape) == 1:
                b_arr = b_arr.reshape(-1, 1)
            filt_tot = np.append(a_arr, b_arr, axis=1)
            w_list_comb = list(product(*w_lists))
            w_list_comb = [np.product(w_here) for w_here in w_list_comb]
            marg_add = [np.array(x_list_comb), w_list_comb, filt_tot]
        else:
            w_list_comb = list(product(*w_lists))
            w_list_comb = [np.product(w_here) for w_here in w_list_comb]
            marg_add = [np.array(x_list_comb), w_list_comb]
            marg_add = reduce_meas(marg_add, filt=filt)

        if all_out == 0:
            return marg_add
        else:
            node_set = list(node_set)
            node_set.sort()
            given_margs[tuple(node_set)] = marg_add
            return given_margs
    elif not rel_par in given_margs:
        if all_out == 0:
            rel_marg = get_marginals_multi(mu, supp_mu, rel_par, g, given_margs, index_mu=index_mu, filt=filt)
        else:
            all_margs = get_marginals_multi(mu, supp_mu, rel_par, g, given_margs, all_out=1, index_mu=index_mu, filt=filt)
            rel_marg = all_margs[rel_par]
    else:
        rel_marg = given_margs[rel_par]
        all_margs = given_margs

    # calculate the joint marginal of node_set given the joint marginal of all the relevant parents:
    d = len(node_set)
    out_x = np.zeros([0, d])
    out_w = []
    if filt == 1:
        out_f = np.zeros([0, 2*d])
    if index_mu == 1:
        if filt == 1:
            supp_rel_p = supp_mu(rel_par, filt=1)
            supp_rel = supp_rel_p[0]
            filt_rel = supp_rel_p[1]
        else:
            supp_rel = supp_mu(rel_par)
    for i in range(len(rel_marg[0])):
        xh = rel_marg[0][i, :]
        wh = rel_marg[1][i]
        if filt == 1:
            fh = rel_marg[2][i, :]  # filtration of marginal; second dimension should be double that of xh
        if wh == 0:
            continue

        if index_mu == 1:
            ind_rel_here = -1
            for j in range(len(supp_rel)):
                if filt == 0:
                    if np.all(np.abs(xh - supp_rel[j, :]) < tol):
                        ind_rel_here = j
                        break
                else:
                    if np.all(np.abs(xh - supp_rel[j, :]) < tol) and np.all(np.abs(fh - filt_rel[j, :]) < tol):
                        ind_rel_here = j
                        break
            if ind_rel_here == -1:
                print('ERROR: relevant support point not found...')

        # for each node in node_set, get the parent values from xh
        x_lists = []
        w_lists = []
        if filt == 1:
            filt_list_a = []
            filt_list_b = []
        for ind, nh in enumerate(node_set):
            if nh in rel_par:
                ind_nh = np.where(nh == rel_par_arr)[0][0]
                x_lists.append([xh[ind_nh]])
                w_lists.append([1])
                if filt == 1:
                    filt_list_a.append([fh[ind_nh]])
                    filt_list_b.append([fh[len(rel_par)+ind_nh]])
            else:
                if len(indices[ind]) > 0:
                    rel_x = xh[indices[ind]]
                    if filt == 1:
                        rel_filt_a = fh[indices[ind]]
                        rel_filt_b = fh[len(rel_par)+indices[ind]]

                else:
                    rel_x = []
                    if filt == 1:
                        rel_filt_a = []
                        rel_filt_b = []
                if index_mu == 0:
                    marg_x_nh = mu(nh, rel_x)
                else:
                    marg_x_nh = mu(nh, ind_rel_here)
                if len(np.array(marg_x_nh[0]).shape) == 2:
                    x_lists.append(marg_x_nh[0][:, 0])  # marg_x_nh[0] is always of shape [n, 1]
                    if filt == 1:
                        filt_list_a.append(marg_x_nh[2][:, 0])
                        filt_list_b.append(marg_x_nh[2][:, 1])
                else:
                    x_lists.append(marg_x_nh[0])
                    if filt == 1:
                        filt_list_a.append(marg_x_nh[2][:, 0])
                        filt_list_b.append(marg_x_nh[2][:, 1])

                w_lists.append(marg_x_nh[1])

        if filt == 1:
            filt_a_comb = list(product(*filt_list_a))
            filt_b_comb = list(product(*filt_list_b))
            a_arr = np.array(filt_a_comb)
            b_arr = np.array(filt_b_comb)
            if len(a_arr.shape) == 1:
                a_arr = a_arr.reshape(-1, 1)
            if len(b_arr.shape) == 1:
                b_arr = b_arr.reshape(-1, 1)
            filt_tot = np.append(a_arr, b_arr, axis=1)
            x_list_comb = list(product(*x_lists))
            w_list_comb = list(product(*w_lists))
            w_list_comb = [np.product(w_here)*wh for w_here in w_list_comb]
            marg_add = [np.array(x_list_comb), w_list_comb, filt_tot]
            out_x = np.append(out_x, marg_add[0], axis=0)
            out_w.extend(marg_add[1])
            out_f = np.append(out_f, filt_tot, axis=0)
        else:
            x_list_comb = list(product(*x_lists))
            w_list_comb = list(product(*w_lists))
            w_list_comb = [np.product(w_here)*wh for w_here in w_list_comb]
            marg_add = [np.array(x_list_comb), w_list_comb]
            marg_add = reduce_meas(marg_add, filt=filt)
            out_x = np.append(out_x, marg_add[0], axis=0)
            out_w.extend(marg_add[1])

    if filt == 1:
        marg_out = [out_x, out_w, out_f]
    else:
        marg_out = [out_x, out_w]
    if all_out == 0:
        return reduce_meas(marg_out, filt=filt)
    else:
        node_set = list(node_set)
        node_set.sort()
        if tuple(node_set) not in all_margs:
            all_margs[tuple(node_set)] = reduce_meas(marg_out, filt=filt)
        for key in given_margs:
            if key not in all_margs:
                all_margs[key] = given_margs[key]
        return all_margs



# FUNCTION TO DIRECTLY SOLVE CAUSAL AND BICAUSAL OT VIA LINEAR PROGRAMMING
def gurobi_bm(margs, f, p_dist=2, radial_cost=0, f_id=0, minmax='min', r_opti=0, outputflag=1, causal=0, anticausal=0):
    """
    :param margs: list with 2 entries, each entry being a discrete probability measure on R^n, where x_list is an [N, n] array
    :param f: function that takes two inputs, x, y, where the inputs are of the form as in the representation of the
    points in margs. Returns a single value
    :param p_dist: if radial cost is used, then this describes the Lp norm which is used.
    :param radial_cost: If 1, then f takes an arbitrary number of inputs but treats them element-wise. Each element
    which will be \|x-y\|_{p_dist} for some x, y. This allows for a faster computation of the cost matrix.
    :param f_id: if non-zero and raidal_cost nonzero, then f will be treated as the identity function.
    :param minmax: if 'min', then we minimize objective, else, we maximize
    :param r_opti: if 0, does not return optimizer. if 1, it does
    :return: optimal value (and optimizer) of the OT problem
    """
    # get relevant data from input:
    m1 = margs[0]
    m2 = margs[1]
    # paths
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    # probability
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1, n_dim = xl_1.shape
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    # build cost matrix:
    # print('Building cost matrix...')
    if radial_cost == 0:
        cost_mat = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                cost_mat[i, j] = f(xl_1[i, :], xl_2[j, :])
    else:
        cost_mat = np.linalg.norm(xl_1[:, None, :] - xl_2[None, :, :], axis=-1, ord=p_dist)
        if f_id == 0:
            cost_mat = f(cost_mat)

    # initialize model
    # print('Initializing model...')
    m = Model('Primal')
    if outputflag == 0:
        m.setParam('OutputFlag', 0)
    pi_var = m.addVars(n1, n2, lb=0, ub=1, name='pi_var')

    # add marginal constraints
    # print('Adding constraints...')
    m.addConstrs((pi_var.sum(i, '*') == pl_1[i] for i in range(n1)), name='first_marg')
    m.addConstrs((pi_var.sum('*', i) == pl_2[i] for i in range(n2)), name='second_marg')

    # add causal constraint: (Note: doesn't seem very efficient, but not sure how else to do)
    causal_count = 0
    if causal == 1:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]
                y_t_arr, ind_inv_y = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]
                    x_tp_arr, ind_inv_p = np.unique(xl_1[pos_h, :t+1], axis=0, return_inverse=True)
                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp = np.where(ind_inv_p == ind_xp)[0]
                        pos_xtp_real = pos_h[pos_xtp]
                        pi_sum_left = 0
                        for i_x in pos_xtp_real:
                            for i_y in pos_h_y:
                                pi_sum_left += pi_var[i_x, i_y]
                        pi_sum_right = 0
                        for i_x in pos_h:
                            for i_y in pos_h_y:
                                pi_sum_right += pi_var[i_x, i_y]
                        mu_sum_left = 0
                        for i_x in pos_h:
                            mu_sum_left += pl_1[i_x]
                        mu_sum_right = 0
                        for i_x in pos_xtp_real:
                            mu_sum_right += pl_1[i_x]

                        causal_count += 1
                        m.addConstr(pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right, name='causal_'+
                                                           str(t)+'_'+str(ind_t)+'_'+str(ind_t_y)+'_'+str(ind_xp))

    if anticausal == 1:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]

                y_t_arr, ind_inv_y = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]

                    x_tp_arr, ind_inv_p = np.unique(xl_2[pos_h, :t+1], axis=0, return_inverse=True)
                    # TODO: note that we have to concatenate pos_h and pos_p to get real index! (done, but good to keep in mind)

                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp = np.where(ind_inv_p == ind_xp)[0]
                        pos_xtp_real = pos_h[pos_xtp]

                        pi_sum_left = 0
                        for i_x in pos_xtp_real:
                            for i_y in pos_h_y:
                                pi_sum_left += pi_var[i_y, i_x]

                        pi_sum_right = 0
                        for i_x in pos_h:
                            for i_y in pos_h_y:
                                pi_sum_right += pi_var[i_y, i_x]

                        mu_sum_left = 0
                        for i_x in pos_h:
                            mu_sum_left += pl_2[i_x]

                        mu_sum_right = 0
                        for i_x in pos_xtp_real:
                            mu_sum_right += pl_2[i_x]

                        m.addConstr(pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right, name='anticausal_'+str(t)+'_'+str(ind_t)+'_'+str(ind_t_y)+'_'+str(ind_xp))

    # Specify objective function
    if minmax == 'min':
        obj = quicksum([cost_mat[i, j] * pi_var[i, j] for i in range(n1) for j in range(n2)])
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        obj = quicksum([cost_mat[i, j] * pi_var[i, j] for i in range(n1) for j in range(n2)])
        m.setObjective(obj, GRB.MAXIMIZE)

    # solve model
    m.optimize()
    objective_val = m.ObjVal

    if r_opti == 0:
        return objective_val
    else:
        return objective_val, [[pi_var[i, j].x for j in range(n2)] for i in range(n1)]

