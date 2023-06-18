import numpy as np
from gurobipy import *
import ot

# src.normal_ot: solve_sinkhorn, gurobi_2d, solve_pot, solve_unbalanced
# This file contains different functions to solve a normal OT problem with two marginals using different methods.

def gurobi_2d(margs, f, p_dist=2, radial_cost=0, f_id=0, minmax='min', r_opti=0, outputflag=1):
    """
    :param margs: list with 2 entries, each entry being a discrete probability measure
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
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1 = len(xl_1)
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    # build cost matrix:
    if radial_cost == 0:
        cost_mat = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                cost_mat[i, j] = f(xl_1[i], xl_2[j])
    else:
        cost_mat = np.linalg.norm(xl_1[:, None, :] - xl_2[None, :, :], axis=-1, ord=p_dist)
        if f_id == 0:
            cost_mat = f(cost_mat)


    # initialize model
    m = Model('Primal')
    if outputflag == 0:
        m.setParam('OutputFlag', 0)
    pi_var = m.addVars(n1, n2, lb=0, ub=1, name='pi_var')

    # add marginal constraints
    m.addConstrs((pi_var.sum(i, '*') == pl_1[i] for i in range(n1)), name='first_marg')
    m.addConstrs((pi_var.sum('*', i) == pl_2[i] for i in range(n2)), name='second_marg')

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


def solve_unbalanced(margs, f, minmax='min', r_opti=0, outputflag=1, epsilon=0.5, alpha=10):
    """
    should converge to normal ot for alpha --> infty and epsilon --> 0.
    In practice, epsilon lower than 0.01 may cause problems
    And on the other hand, larger values of epsilon and lower values of alpha converge a lot faster
    Notably, the penalization by alpha allows for more general couplings
    (that do not have to satisfy marginal constraints)
    On the other hand, regularization by epsilon simply restricts the couplings (by requiring smoothness)
    Hence for low values of alpha, optimal value is usually above the true OT
    And for high value sof epsilon, optimal value is usually below the true OT
    """
    m1 = margs[0]
    m2 = margs[1]
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1 = len(xl_1)
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    cost_mat = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = f(xl_1[i], xl_2[j])
    if minmax == 'max':
        cost_mat *= -1
    Gs = ot.unbalanced.sinkhorn_unbalanced(pl_1, pl_2, cost_mat, epsilon, alpha, verbose=outputflag)

    if r_opti == 0:
        return np.sum(Gs*cost_mat)
    else:
        return np.sum(Gs*cost_mat), Gs


def solve_pot(margs, f, minmax='min', r_opti=0, outputflag=1):
    m1 = margs[0]
    m2 = margs[1]
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1 = len(xl_1)
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    cost_mat = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = f(xl_1[i], xl_2[j])
    if minmax == 'max':
        cost_mat *= -1
    Gs = ot.emd(pl_1, pl_2, cost_mat)

    if r_opti == 0:
        return np.sum(Gs*cost_mat)
    else:
        return np.sum(Gs*cost_mat), Gs


def solve_sinkhorn(margs, f, minmax='min', r_opti=0, outputflag=1, epsilon=0.1):
    # should converge to normal ot for epsilon --> 0.
    # In practice, epsilon lower than around 0.01 may cause problems
    m1 = margs[0]
    m2 = margs[1]
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1 = len(xl_1)
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    cost_mat = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = f(xl_1[i], xl_2[j])
    if minmax == 'max':
        cost_mat *= -1
    Gs = ot.sinkhorn(pl_1, pl_2, cost_mat, epsilon)

    if r_opti == 0:
        return np.sum(Gs*cost_mat)
    else:
        return np.sum(Gs*cost_mat), Gs



def make_new_vf(p_mu, p_nu, vals):
    # program used to create a new value function out of the solutions to the DPP problems at a given node
    n_vf = len(p_nu)

    def new_vf(x, y):
        ind1 = np.where((np.abs(p_mu - x) <= 10 ** -6).all(axis=1))
        ind1 = ind1[0][0]
        ind2 = np.where((np.abs(p_nu - y) <= 10 ** -6).all(axis=1))
        ind2 = ind2[0][0]
        return vals[ind1 * n_vf + ind2]

    return new_vf


def solve_dynamic(cost, mu, nu, supp_mu, supp_nu, g, index_mu=0, index_nu=0, outputflag=1, method='gurobi'):
    """
    should solve a (graph) causal Wasserstein model using dynamic programming
    :param cost: a list of functions. each function contains two elements. first element is a list of nodes that this
    entry depends on. Second is a function that takes as input a tuple of support points of the relevant nodes and
    returns a value. The "true" cost function of the OT problem is the sum of all individual costs.
    :param mu: first marginal. see DAGmeasures.py regarding structure
    :param nu: same as mu
    :param supp_mu: support of first marginal. See DAGmeasures.py regarding structure
    :param supp_nu: same as supp_mu
    :param g: the graph structure
    :param method: 'gurobi', 'pot', 'sinkhorn', 'unbalanced' specifies which method is used to solve OT problems
    :return:
    """
    out_vals = []
    ordering = g.topologicalSort()
    T = g.V
    cur_V = cost  # current list of relevant value functions
    # each element of cur_V is a list with two entries. The first entry is a list of nodes that this value function
    # depends on, and the second entry is a function which returns a value for each tuple of support points of the
    # respective nodes

    optis_nodes = []  # list with T entries
    rel_node_list = []  # list with T entries, each being a tuple of the relevant nodes for the current node
    optis_pars = []  # list with T entries, each being an [n_1, 2*d] array of points
    optis_meas = []  # list with T entries, each being a list of n_1 entries,
    # each being a list with two entries, one [n_2, 2] array of points and one list of length
    # n_2 with weights

    for i_t in range(T):
        # get current node that we work backwards from
        v_here = ordering[T - 1 - i_t]
        optis_nodes.append(v_here)

        # get relevant nodes that are in connection with current node
        rel_nodes = g.parents[v_here].copy()

        # get relevant nodes and value functions that are in connection with current node
        rel_value_funs = []
        cur_V_old = cur_V.copy()
        cur_V = []
        for v_old in cur_V_old:
            if v_here in v_old[0]:
                rel_nodes.extend(v_old[0])
                rel_value_funs.append(v_old)
            else:
                cur_V.append(v_old)

        # relevant nodes should only include all connected nodes, but not the current node, sorted
        rel_nodes = list(set(rel_nodes))
        if v_here in rel_nodes:
            rel_nodes.remove(v_here)
        n_rel = len(rel_nodes)
        rel_nodes.sort()
        rel_node_list.append(tuple(rel_nodes))

        # # get all parent indices ... I don't think I need this after all.
        # par_indices = []
        # for v_par in g.parents[v_here]:
        #     par_indices.append(rel_nodes.index(v_par))
        # par_indices = np.array(par_indices)
        par_indices = g.parents[v_here]

        # make another array of all relevant nodes including the current node, sorted
        rel_plus_cur_nodes = rel_nodes.copy()
        rel_plus_cur_nodes.append(v_here)
        rel_plus_cur_nodes.sort()

        # get relevant supports of all measures involved:
        supp_h_mu = supp_mu(rel_nodes)  # array of size N_h_mu x n_rel
        supp_h_nu = supp_nu(rel_nodes)  # array of size N_h_nu x n_rel
        N_h_mu = len(supp_h_mu)
        N_h_nu = len(supp_h_nu)

        # if the second dimension is for some reason dimension zero, this basically still means that there are no relevant parents
        if np.prod(np.array(supp_h_mu).shape) == 0:
            N_h_mu = 0
        if np.prod(np.array(supp_h_nu).shape) == 0:
            N_h_nu = 0

        # iterate over pairs of support points of the related nodes
        vals = []
        optis = []

        # look at the case where both mu and nu have no relevant parents
        if N_h_nu == N_h_mu == 0:
            # for each pair of support points, get the disintegrations (i.e. measures) that are relevant for the
            # OT problem
            if index_mu == 1:
                input_mu = mu(v_here, 0)
            else:
                input_mu = mu(v_here, [])
            if index_nu == 1:
                input_nu = nu(v_here, 0)
            else:
                input_nu = nu(v_here, [])

            # for each pair of support points, build the cost function out of the rel_value_funs for the OT problem
            def input_fun(x, y):
                out = 0
                for vf in rel_value_funs:
                    out += vf[1](x[0:1], y[0:1])
                return out

            # solve OT problem!
            ov, opti = gurobi_2d([input_mu, input_nu], input_fun, r_opti=1, outputflag=outputflag)
            out_vals.append(ov)
            vals.append(ov)

            # set optimizer:
            optis_par = np.zeros([1, 0])  # no parent values
            pmu = input_mu[0]
            pnu = input_nu[0]
            nmu = len(pmu)
            nnu = len(pnu)
            optis_x = np.zeros([nmu * nnu, 2])
            optis_w = np.zeros(nmu * nnu)
            for i in range(len(pmu)):
                for j in range(len(pnu)):
                    optis_x[i + nmu * j, 0] = pmu[i]
                    optis_x[i + nmu * j, 1] = pnu[j]
                    optis_w[i + nmu * j] = opti[i][j]

            optis_pars.append(optis_par)
            optis_meas.append([[optis_x, optis_w]])

        else:
            optis_par = np.zeros([N_h_nu * N_h_mu, 2 * n_rel])
            optis_meas_h = []
            for i in range(N_h_mu):
                for j in range(N_h_nu):

                    # get the x and y point that is relevant for this iteration
                    p_mu_h = supp_h_mu[i, :]
                    p_nu_h = supp_h_nu[j, :]
                    p_mu_h_ext = np.zeros(T)
                    p_nu_h_ext = np.zeros(T)
                    p_mu_h_ext[rel_nodes] = p_mu_h
                    p_nu_h_ext[rel_nodes] = p_nu_h

                    # extract parents for disintegration
                    p_par_mu_h = p_mu_h_ext[par_indices]
                    p_par_nu_h = p_nu_h_ext[par_indices]

                    # for each pair of support points, get the disintegrations (i.e. measures) that are relevant for the
                    # OT problem
                    input_mu = mu(v_here, p_par_mu_h)
                    input_nu = nu(v_here, p_par_nu_h)

                    # for each pair of support points, build cost function out of the rel_value_funs for the OT problem
                    def input_fun(x, y):
                        out = 0
                        p_mu_h_ext_vf = p_mu_h_ext.copy()
                        p_nu_h_ext_vf = p_nu_h_ext.copy()
                        p_mu_h_ext_vf[v_here] = x
                        p_nu_h_ext_vf[v_here] = y

                        for vf in rel_value_funs:
                            inds_vf = vf[0]
                            xinpvf = p_mu_h_ext_vf[inds_vf]
                            yinpvf = p_nu_h_ext_vf[inds_vf]
                            out += vf[1](xinpvf, yinpvf)
                        return out

                    # solve OT problem!
                    if method == 'gurobi':
                        ov, opti = gurobi_2d([input_mu, input_nu], input_fun, r_opti=1, outputflag=outputflag)
                    elif method == 'unbalanced':
                        ov, opti = solve_unbalanced([input_mu, input_nu], input_fun, r_opti=1, outputflag=outputflag)
                    elif method == 'pot':
                        ov, opti = solve_pot([input_mu, input_nu], input_fun, r_opti=1, outputflag=outputflag)
                    elif method == 'sinkhorn':
                        ov, opti = solve_sinkhorn([input_mu, input_nu], input_fun, r_opti=1, outputflag=outputflag)

                    vals.append(ov)
                    optis.append(opti)
                    opti = np.array(opti)

                    # set optis_x:
                    optis_par[i * N_h_nu + j, :n_rel] = p_mu_h
                    optis_par[i * N_h_nu + j, n_rel:] = p_nu_h
                    pmu = input_mu[0]
                    pnu = input_nu[0]
                    nmu = len(pmu)
                    nnu = len(pnu)
                    optis_x = np.zeros([nmu * nnu, 2])
                    optis_w = np.zeros(nmu * nnu)
                    for ii in range(len(pmu)):
                        for jj in range(len(pnu)):
                            optis_x[ii + nmu * jj, 0] = pmu[ii]
                            optis_x[ii + nmu * jj, 1] = pnu[jj]
                            optis_w[ii + nmu * jj] = opti[ii, jj]
                    optis_meas_h.append([optis_x, optis_w])
            optis_pars.append(optis_par)
            optis_meas.append(optis_meas_h)

        # build new value function out of solutions to OT problems and add it to cur_V
        new_vf = make_new_vf(supp_h_mu, supp_h_nu, vals)
        V_new = [rel_nodes, new_vf]
        cur_V.append(V_new)

        # save optimal coupling in some suitable fashion...

    # at the end, cur_V should only contain one function that does not depend on any input. The output value is the
    # returns value

    return out_vals, [optis_nodes, rel_node_list, optis_pars, optis_meas]

