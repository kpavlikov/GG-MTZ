#from itertools import product
import numpy as np
from mip import Model, xsum, BINARY, maximize, OptimizationStatus, GRB
from utilities import min_capacity_cover_for_S, create_HCVRP_CI, bin_pack_S
from time import *
#=======================================================================================================================
def create_strong_FCI(HCVRP_model, x, a, S):
    C = HCVRP_model._C
    K = HCVRP_model._K
    Q = HCVRP_model._Q
    mu = HCVRP_model._mu
    demand = HCVRP_model._demand
    y = {}
    z = {}
    if len(S) < HCVRP_model._n:
        HCVRP_model._FCI_counter += 1
    cut_number = HCVRP_model._FCI_counter
    required_capacity = min_capacity_cover_for_S(S, Q, None, mu, demand)
    set_K = set(range(0, K))
    if S == C:
        mu_tilde_S = mu
    else:
        mu_tilde_S = [0 for k in set_K]
        for k in set_K:
            mu_tilde_S[k] = min(bin_pack_S(k, S, Q, demand), mu[k])
    mC = np.zeros((HCVRP_model._K, max(mu_tilde_S) + 1))
    for k in set_K:
        C_vector = min_capacity_cover_for_S(S, Q, k, mu_tilde_S, demand)
        mC[k, 0] = C_vector[0]
        for e in range(1, mu_tilde_S[k] + 1):
            mC[k, e] = C_vector[e] - C_vector[0]
    set_K_S = set()
    for k in set_K:
        if required_capacity + 1 <= mC[k, 0] or (abs(required_capacity - mC[k, 0]) < 0.001 and max(mC[k, l] for l in range(1, mu_tilde_S[k] + 1)) >= 1):
            set_K_S.add(k)
            for e in range(1, mu_tilde_S[k] + 1):
                y[cut_number, k, e] = HCVRP_model.add_var()
            HCVRP_model += xsum(y[cut_number, k, e] for e in range(1, mu_tilde_S[k] + 1)) <= 1
    if S == C:
        for k in set_K_S:
            HCVRP_model += xsum(a[i, k] for i in C) == xsum(e * y[cut_number, k, e] for e in range(1, mu_tilde_S[k] + 1))
        for k in set_K - set_K_S:
            z[cut_number, k] = 0
        min_number_of_vehicles_for_S = xsum(a[i, k] for i in C for k in set_K)
    else:
        for k in set_K_S:
            HCVRP_model += xsum(e * y[cut_number, k, e] for e in range(1, mu_tilde_S[k]+1)) >= xsum(a[i, k] for i in S)
            HCVRP_model += xsum(e * y[cut_number, k, e] for e in range(1, mu_tilde_S[k]+1)) <= xsum(a[i, k] for i in C)
        for k in set_K - set_K_S:
            z[cut_number, k] = HCVRP_model.add_var()
            HCVRP_model += z[cut_number, k] <= xsum(a[i, k] for i in C - S)
        min_number_of_vehicles_for_S = xsum(e * y[cut_number, k, e] for k in set_K_S for e in range(1, mu_tilde_S[k] + 1)) + \
                                       xsum(z[cut_number, k] + xsum(a[i, k] for i in S) for k in set_K - set_K_S)
    #print(len(set_K), len(set_K_S))

    HCVRP_model += xsum(Q[r] * (z[cut_number, r] + xsum(a[i, r] for i in S)) for r in set_K - set_K_S) + \
                   xsum(Q[r] * e * y[cut_number, r, e] for r in set_K_S for e in range(1, mu_tilde_S[r] + 1)) \
                   >= \
                   required_capacity
    if len(set_K_S) > 0:
        for k in set_K_S:
            #print(required_capacity, k, mC[k, :])
            HCVRP_model += xsum(Q[r] * (z[cut_number, r] + xsum(a[i, r] for i in S)) for r in set_K - set_K_S) + \
                           xsum(Q[r] * e * y[cut_number, r, e] for r in set_K_S - {k} for e in range(1, mu_tilde_S[r] + 1)) + \
                           xsum((Q[k] * e - mC[k, e]) * y[cut_number, k, e] for e in range(1, mu_tilde_S[k] + 1)) >= \
                           mC[k, 0]
    Gomory_cut_counter = 0
    for k in set_K - set_K_S:
        Gomory_cut_counter += 1
        HCVRP_model += z[cut_number, k] + xsum(a[i, k] for i in S) + \
                       xsum(np.ceil(Q[r] / Q[k]) * (z[cut_number, r] + xsum(a[i, r] for i in S)) for r in set_K - set_K_S - {k}) + \
                       xsum(xsum(np.ceil(Q[r] * e / Q[k]) * y[cut_number, r, e] for e in range(1, mu_tilde_S[r] + 1)) for r in set_K_S) \
                       >= \
                       np.ceil(required_capacity / Q[k]), 'FCI_'+str(cut_number)+'_'+str(Gomory_cut_counter)
    for k in set_K_S:
        for t in set_K_S - {k}:
            for q in range(1, mu_tilde_S[t]+1):
                Gomory_cut_counter += 1
                coeff = Q[t] * q
                HCVRP_model += xsum(np.ceil(Q[r] / coeff) * (z[cut_number, r] + xsum(a[i, r] for i in S)) for r in set_K - set_K_S) + \
                                y[cut_number, t, q] + xsum(np.ceil(e / q) * y[cut_number, t, e] for e in range(1, mu_tilde_S[t] + 1) if e != q) + \
                                xsum(xsum(np.ceil(Q[r] * e / coeff) * y[cut_number, r, e] for e in range(1, mu_tilde_S[r] + 1)) for r in set_K_S - {t, k}) + \
                                xsum(np.ceil((Q[k] * e - mC[k, e]) / coeff) * y[cut_number, k, e] for e in range(1, mu_tilde_S[k] + 1)) \
                                >= \
                                np.ceil(mC[k, 0] / coeff), 'FCI_'+str(cut_number)+'_'+str(Gomory_cut_counter)
        for e in range(1, mu_tilde_S[k] + 1):
            coeff = Q[k] * e - mC[k, e]
            coeff_abs = abs(coeff)
            if coeff_abs > 1:
                if coeff > 0:
                    sign = 1
                else:
                    sign = -1
                Gomory_cut_counter += 1
                HCVRP_model += xsum(np.ceil(Q[r] / coeff_abs) * (z[cut_number, r] + xsum(a[i, r] for i in S)) for r in set_K - set_K_S) + \
                                xsum(xsum(np.ceil(Q[r] * q / coeff_abs) * y[cut_number, r, q] for q in range(1, mu_tilde_S[r] + 1)) for r in set_K_S - {k}) + \
                                sign * y[cut_number, k, e] + xsum(np.ceil((Q[k] * q - mC[k, q]) / coeff_abs) * y[cut_number, k, q] for q in range(1, mu_tilde_S[k] + 1) if e != q) \
                                >= \
                                np.ceil(mC[k, 0] / coeff_abs), 'FCI_'+str(cut_number)+'_'+str(Gomory_cut_counter)
    cut_LHS, cut_RHS = create_HCVRP_CI(x, S, C, min_number_of_vehicles_for_S)
    HCVRP_model += cut_LHS <= cut_RHS, 'FCI_'+str(cut_number)
    print('Imposed Gomory cuts: ', Gomory_cut_counter)
#=======================================================================================================================
def FCI_Sep(HCVRP_model, x, a):
    epsilon_1 = HCVRP_model._epsilon_1
    epsilon_2 = HCVRP_model._epsilon_2
    C = HCVRP_model._C
    K = HCVRP_model._K
    Q = HCVRP_model._Q
    demand = HCVRP_model._demand
    vehicles_types_used = np.zeros(K)

    #FCI_sep_model = Model(solver_name=CBC)
    FCI_sep_model = Model(solver_name=GRB)
    FCI_sep_model.verbose = 0
    delta = {}
    gamma = {}
    theta = {}
    beta = {}
    for k in range(0, K):
        beta[k] = FCI_sep_model.add_var(var_type=BINARY)
        theta[k] = FCI_sep_model.add_var()
        FCI_sep_model += theta[k] <= 1 - beta[k]
    for k in range(0, K - 1):
        FCI_sep_model += beta[k] + theta[k] >= beta[k + 1] + theta[k + 1]
    FCI_sep_model.add_sos([(theta[k], 1) for k in range(0, K)], 1)
    for i in C:
        delta[i] = FCI_sep_model.add_var(var_type=BINARY)
    FCI_sep_model += xsum(delta[i] for i in C) >= 2

    for (i, j) in x.keys():
        if i > 0 and i < j:
            gamma[i, j] = FCI_sep_model.add_var()
            FCI_sep_model += gamma[i, j] <= delta[i]
            FCI_sep_model += gamma[i, j] <= delta[j]

    start_time = time()
    print()
    print('Launching FCI separation...')
    need_to_run_FCI_separation = True
    use_incumbent_solutions = True
    while need_to_run_FCI_separation:
        status = HCVRP_model.optimize()
        print('HCVRP objective: ', round(HCVRP_model.objective_value, 2), 'CPU: ', round(time() - start_time, 1))
        if status == OptimizationStatus.OPTIMAL:
            for k in range(0, K):
                vehicles_types_used[k] = sum(a[i, k].x for i in C)
            if FCI_sep_model.constr_by_name('FCI_main'):
                FCI_sep_model.remove(FCI_sep_model.constr_by_name('FCI_main'))
            FCI_sep_model += xsum(Q[k] * vehicles_types_used[k] * (beta[k] + theta[k]) for k in range(0, K)) <= xsum(delta[i] * demand[i] for i in C), 'FCI_main'
            observed_capacity = xsum(delta[i] * (x[0, i].x + x[i, 0].x) for i in C if (x[0, i].x + x[i, 0].x) >= epsilon_1) + xsum((delta[i] + delta[j] - 2 * gamma[i, j]) * (x[j, i].x + x[i, j].x) for (i, j) in gamma.keys() if (x[j, i].x + x[i, j].x) >= epsilon_1)

            FCI_sep_model.objective = maximize(2 * xsum((beta[k] + theta[k]) * vehicles_types_used[k] for k in range(0, K)) - observed_capacity)
            FCI_sep_model.optimize()
            #print('Separation solved, d(S) = ', sum(delta[i].x * demand[i] for i in C))
            #print('observed capacity ', round(observed_capacity.x, 2))
            #print('target capacity ', round(2 * sum((beta[k].x + alpha[k].x) * vehicles_types_used[k] for k in range(0, K)), 2))
            #print('violation ', round(FCI_sep_model.objective_value, 2))
            if FCI_sep_model.objective_value >= epsilon_2:
                if use_incumbent_solutions:
                    for q in range(0, FCI_sep_model.num_solutions):
                        if FCI_sep_model.objective_values[q] >= epsilon_2:
                            S = {i for i in HCVRP_model._C if delta[i].xi(q) > 0.5}
                            print('Adding violated cut, violation = ', round(FCI_sep_model.objective_values[q], 3))
                            create_strong_FCI(HCVRP_model, x, a, S)
                else:
                    S = {i for i in HCVRP_model._C if delta[i].x > 0.5}
                    print('Adding violated cut, violation = ', round(FCI_sep_model.objective_value, 3))
                    create_strong_FCI(HCVRP_model, x, a, S)
                print('Total FCI cuts added: ', HCVRP_model._FCI_counter)
            else:
                need_to_run_FCI_separation = False
#=======================================================================================================================