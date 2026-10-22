#from itertools import product
from mip import Model, xsum, BINARY, INTEGER, maximize, OptimizationStatus, GRB
from time import *
from utilities import create_HCVRP_CI
#=======================================================================================================================
def find_min_number_vehicles(S, Q, mu, demand):
    S_demand = sum(demand[i] for i in S)
    find_min_number_vehicles = 1
    for k in range(0, len(Q)):
        #print(k, mu[k])
        for t in range(0, mu[k]):
            S_demand = S_demand - Q[k]
            #print(t, Q[k], S_demand)
            if S_demand <= 0:
                #find_min_number_vehicles += 1
                break
            else:
                find_min_number_vehicles += 1
    #print(sum(demand[i] for i in S), find_min_number_vehicles)
    return find_min_number_vehicles
#=======================================================================================================================
def create_RCI(x, S, C, Q, mu, demand):
    min_number_of_vehicles_for_S = find_min_number_vehicles(S, Q, mu, demand)

    cut_LHS, cut_RHS = create_HCVRP_CI(x, S, C, min_number_of_vehicles_for_S)
    return cut_LHS, cut_RHS
#=======================================================================================================================
def RCI_Sep(model, x):
    epsilon_1 = model._epsilon_1
    epsilon_2 = model._epsilon_2
    K = model._K
    Q = model._Q
    C = model._C
    mu = model._mu
    demand = model._demand

    #RCI_sep_model = Model(solver_name=CBC)
    RCI_sep_model = Model(solver_name=GRB)
    RCI_sep_model.verbose = 0
    delta = {}
    alpha = {}
    beta = {}
    gamma = {}
    for i in C:
        # binary variables indicating whether i set S
        delta[i] = RCI_sep_model.add_var(var_type=BINARY)
    RCI_sep_model += xsum(delta[i] for i in C) >= 2

    for (i, j) in x.keys():
        if i > 0 and i < j:
            gamma[i, j] = RCI_sep_model.add_var()
            RCI_sep_model += gamma[i, j] <= delta[i]
            RCI_sep_model += gamma[i, j] <= delta[j]
    if mu is not None:
        for k in range(0, K):
            beta[k] = RCI_sep_model.add_var(var_type=BINARY)
            alpha[k] = RCI_sep_model.add_var(var_type=INTEGER)
            RCI_sep_model += alpha[k] <= mu[k] * (1 - beta[k])
        for k in range(0, K - 1):
            RCI_sep_model += beta[k] + alpha[k] / mu[k] >= beta[k + 1] + alpha[k + 1] / mu[k + 1]
        RCI_sep_model.add_sos([(alpha[k], 1) for k in range(0, K)], 1)
        RCI_sep_model += xsum(Q[k] * (beta[k] * mu[k] + alpha[k]) for k in range(0, K)) + 1 <= xsum(delta[i] * demand[i] for i in C)
    else:
        alpha[0] = RCI_sep_model.add_var(var_type=INTEGER)
        RCI_sep_model += Q[0] * alpha[0] + 1 <= xsum(delta[i] * demand[i] for i in C)

    start_time = time()
    print()
    print('Launching RCI separation...')
    need_to_run_RCC_separation = True
    use_incumbent_solutions = True
    while need_to_run_RCC_separation:
        status = model.optimize()
        print('HCVRP objective: ', round(model.objective_value, 2), 'CPU: ', round(time() - start_time, 2))
        if status == OptimizationStatus.OPTIMAL:
            observed_capacity = xsum(delta[i] * (x[0, i].x + x[i, 0].x) for i in C if (x[0, i].x + x[i, 0].x) >= epsilon_1) + \
                                xsum((delta[i] + delta[j] - 2 * gamma[i, j]) * (x[i, j].x + x[j, i].x) for (i, j) in gamma.keys() if (x[i, j].x + x[j, i].x) >= epsilon_1)
            if mu is not None:
                RCI_sep_model.objective = maximize(2 + 2 * xsum((beta[k] * mu[k] + alpha[k]) for k in range(0, K)) - observed_capacity)
            else:
                RCI_sep_model.objective = maximize(2 + 2 * alpha[0] - observed_capacity)
        RCI_sep_model.optimize()
        #print('Separation solved, d(S) = ', sum(delta[i].x * demand[i] for i in C))
        #print('observed capacity ', round(observed_capacity.x, 2))
        #if available_vehicles is not None:
        #    print('target capacity ', round(2 + 2 * sum(beta[k].x * available_vehicles[k] + alpha[k].x for k in range(0, K)), 2))
        #else:
        #    print('target capacity ', round(2 + 2 * alpha[0].x, 2))
        #print('violation ', np.round(RCI_sep_model.objective_value, 2))
        #remaining_demand = sum(delta[i].x * demand[i] for i in C)
        #if available_vehicles is not None:
        #    for k in range(0, K):
        #        remaining_demand -= available_vehicles[k] * Q[k] * beta[k].x + alpha[k].x * Q[k]
        #        if beta[k].x + alpha[k].x >= epsilon_1:
        #            print(k, round(beta[k].x), round(alpha[k].x, 2), round(available_vehicles[k], 2), round(remaining_demand, 2))
        if RCI_sep_model.objective_value >= epsilon_2:
            if use_incumbent_solutions:
                for q in range(0, RCI_sep_model.num_solutions):
                    if RCI_sep_model.objective_values[q] >= epsilon_2:
                        S = {i for i in C if delta[i].xi(q) > 0.5}
                        print('Adding violated cut, violation = ', round(RCI_sep_model.objective_values[q], 3))
                        # print(model.num_solutions)
                        # print(model.objective_value[k], {i for i in C if delta[i].xi(k) > 0.5})
                        model._RCI_counter += 1
                        cut_LHS, cut_RHS = create_RCI(x, S, C, Q, mu, demand)
                        model += cut_LHS <= cut_RHS, 'RCI_'+str(model._RCI_counter)
            else:
                S = {i for i in C if delta[i].x > 0.5}
                print('Adding violated cut, violation = ', round(RCI_sep_model.objective_value, 3))
                model._RCI_counter += 1
                cut_LHS, cut_RHS = create_RCI(x, S, C, Q, mu, demand)
                model += cut_LHS <= cut_RHS, 'RCI_'+str(model._RCI_counter)
            print('Total RCI added: ', model._RCI_counter)
        else:
            need_to_run_RCC_separation = False
#=======================================================================================================================