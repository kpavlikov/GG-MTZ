#from itertools import product
import numpy as np
from mip import Model, xsum, BINARY, INTEGER, maximize, OptimizationStatus, GRB
from time import *
from utilities import create_HCVRP_CI
#=======================================================================================================================
def create_BBV_RCI(k, x, a, S, C, Q, demand):
    min_number_of_vehicles_for_S = np.ceil(sum(demand[i] for i in S) / Q[k + 1]) - xsum(np.ceil(Q[h] / Q[k + 1] - 1) * a[i, h] for h in range(0, k + 1) for i in C)
    cut_LHS, cut_RHS = create_HCVRP_CI(x, S, C, min_number_of_vehicles_for_S)
    return cut_LHS, cut_RHS
#=======================================================================================================================
def BBV_RCI_Sep(model, x, a):
    epsilon_1 = model._epsilon_1
    epsilon_2 = model._epsilon_2

    #BBV_RCI_sep_model = Model(solver_name=CBC)
    BBV_RCI_sep_model = Model(solver_name=GRB)
    BBV_RCI_sep_model.verbose = 0
    delta = {}
    gamma = {}
    alpha = {}
    C = model._C
    K = model._K
    Q = model._Q
    demand = model._demand
    for i in C:
        # binary variable delta_i indicating whether i is in set S
        delta[i] = BBV_RCI_sep_model.add_var(var_type=BINARY)
    BBV_RCI_sep_model += xsum(delta[i] for i in C) >= 2

    for (i, j) in x.keys():
        if i > 0 and i < j:
            gamma[i, j] = BBV_RCI_sep_model.add_var()
            BBV_RCI_sep_model += gamma[i, j] <= delta[i]
            BBV_RCI_sep_model += gamma[i, j] <= delta[j]

    start_time = time()
    print()
    print('Launching BBV RCI separation...')
    need_to_run_BBV_RCI_sep = True
    use_incumbent_solutions = True
    while need_to_run_BBV_RCI_sep:
        status = model.optimize()
        print('HCVRP objective: ', round(model.objective_value, 2), 'CPU: ', round(time() - start_time, 1))
        if status == OptimizationStatus.OPTIMAL:
            need_to_run_BBV_RCI_sep = False
            observed_capacity = xsum(delta[i] * (x[0, i].x + x[i, 0].x) for i in C if (x[0, i].x + x[i, 0].x) >= epsilon_1) + \
                                xsum((delta[i] + delta[j] - 2 * gamma[i, j]) * (x[i, j].x + x[j, i].x) for (i, j) in gamma.keys() if (x[i, j].x + x[j, i].x) >= epsilon_1)
            for k in range(0, K - 1):
                alpha[k + 1] = BBV_RCI_sep_model.add_var(var_type=INTEGER)
                BBV_RCI_sep_model += Q[k + 1] * alpha[k + 1] + 1 <= xsum(delta[i] * demand[i] for i in C)
                BBV_RCI_sep_model.objective = maximize(2 + 2 * alpha[k + 1] - 2 * xsum(np.ceil((Q[h] / Q[k + 1] - 1)) * a[i, h].x for i in model._C for h in range(0, k + 1)) - observed_capacity)
                BBV_RCI_sep_model.optimize()
                if BBV_RCI_sep_model.objective_value >= epsilon_2:
                    #remaining_demand = sum(delta[i].x * demand[i] for i in C)
                    #print('Separation solved, d(S) = ', remaining_demand)
                    #print('observed capacity ', round(observed_capacity.x, 2))
                    if use_incumbent_solutions:
                        for sol_number in range(0, BBV_RCI_sep_model.num_solutions):
                            if BBV_RCI_sep_model.objective_values[sol_number] >= epsilon_2:
                                S = {i for i in C if delta[i].xi(sol_number) > 0.5}
                                print('Adding violated cut, violation = ', round(BBV_RCI_sep_model.objective_values[sol_number], 3))
                                model._BBV_counter += 1
                                cut_LHS, cut_RHS = create_BBV_RCI(k, x, a, S, C, Q, demand)
                                model += cut_LHS <= cut_RHS, 'BBV_'+str(model._BBV_counter)
                    else:
                        S = {i for i in C if delta[i].x > 0.5}
                        print('Adding violated cut, violation = ', round(BBV_RCI_sep_model.objective_value, 3))
                        model._BBV_counter += 1
                        cut_LHS, cut_RHS = create_BBV_RCI(k, x, a, S, C, Q, demand)
                        model += cut_LHS <= cut_RHS, 'BBV_'+str(model._BBV_counter)
                    need_to_run_BBV_RCI_sep = True
            print('Total BBV_RCI cuts added: ', model._BBV_counter)
#=======================================================================================================================