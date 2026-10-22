from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, minimize, GRB
import numpy as np
from itertools import product
#-----------------------------------------------------------------------------------------------------------------------
def make_binary(x):
    for element in list(x.keys()):
        x[element].__setattr__('var_type', BINARY)
#-----------------------------------------------------------------------------------------------------------------------
def bin_pack_S(k, S, Q, demand):
    #bin_pack_model = Model(solver_name=CBC)
    bin_pack_model = Model(solver_name=GRB)
    bin_pack_model.verbose = 0
    delta = {}
    ind = {}
    max_number_of_bins = len({i for i in S if demand[i] <= Q[k]})
    for t in range(0, max_number_of_bins):
        ind[t] = bin_pack_model.add_var(var_type=BINARY)
        for i in S:
            if demand[i] <= Q[k]:
                delta[t, i] = bin_pack_model.add_var(var_type=BINARY)
        bin_pack_model += xsum(demand[i] * delta[t, i] for i in S if demand[i] <= Q[k]) <= Q[k] * ind[t]
    for i in S:
        if demand[i] <= Q[k]:
            bin_pack_model += xsum(delta[t, i] for t in range(0, max_number_of_bins)) == 1
    bin_pack_model.objective = minimize(xsum(ind[t] for t in range(0, max_number_of_bins)))
    bin_pack_model.optimize()
    obj = int(round(bin_pack_model.objective_value))
    return obj
#=======================================================================================================================
def min_capacity_cover_for_S(S, Q, k, mu, demand):
    #min_capacity_model = Model(solver_name=CBC)
    min_capacity_model = Model(solver_name=GRB)
    min_capacity_model.verbose = 0
    z = {}
    for r in range(0, len(Q)):
        z[r] = min_capacity_model.add_var(var_type=INTEGER, ub=mu[r])
    min_capacity_model += xsum(Q[r] * z[r] for r in range(0, len(Q))) >= sum(demand[i] for i in S)
    min_capacity_model.objective = minimize(xsum(Q[r] * z[r] for r in range(0, len(Q))))
    min_capacity_model.optimize()
    min_capacity_cover_for_S = min_capacity_model.objective_value
    if k is not None:
        min_capacity_cover_for_S = np.zeros(mu[k] + 1)
        for e in range(0, mu[k] + 1):
            if e > 0:
                min_capacity_model.remove(min_capacity_model.constr_by_name('tmp'))
            min_capacity_model += z[k] == e, 'tmp'
            min_capacity_model.optimize()
            min_capacity_cover_for_S[e] = min_capacity_model.objective_value
    return min_capacity_cover_for_S
#=======================================================================================================================
def create_HCVRP_CI(x, S, C, target_number_of_vehicles):
    len_S = len(S)
    n = len(C)
    if len_S <= n * (n + 1) / (2 * n - 2):
        cut_LHS = xsum(x[i, j] for (i, j) in product(S, S) if i != j)
        cut_RHS = len_S - target_number_of_vehicles
    elif len_S == n:
        cut_LHS = 2 * target_number_of_vehicles
        cut_RHS = xsum(x[0, i] + x[i, 0] for i in C)
    else:
        S_bar = C.difference(S)
        cut_LHS = xsum(x[i, j] for (i, j) in product(S_bar, S_bar) if i != j) + 0.5 * xsum(x[0, i] + x[i, 0] for i in S_bar) - 0.5 * xsum(x[0, i] + x[i, 0] for i in S)
        cut_RHS = n - len_S - target_number_of_vehicles
    return cut_LHS, cut_RHS
#=======================================================================================================================
def remove_nonbinding_RCI(model):
    cuts_to_be_removed = ['RCI_' + str(cut_number) for cut_number in range(1, model._RCI_counter + 1) \
                          if abs(model.constr_by_name('RCI_' + str(cut_number)).expr.x) >= model._epsilon_3]
    number_removed_constraints = len(cuts_to_be_removed)
    model.remove([model.constr_by_name(code) for code in cuts_to_be_removed])
    print('Removed nonbinding RCI cuts: ', number_removed_constraints, ' Remaining RCI: ', model._RCI_counter - number_removed_constraints)
    #model.remove(RCI_cuts_to_remove)
#=======================================================================================================================
def remove_nonbinding_BBV(model):
    cuts_to_be_removed = ['BBV_' + str(cut_number) for cut_number in range(1, model._BBV_counter + 1) \
                           if abs(model.constr_by_name('BBV_' + str(cut_number)).expr.x) >= model._epsilon_3]
    number_removed_constraints = len(cuts_to_be_removed)
    model.remove([model.constr_by_name(code) for code in cuts_to_be_removed])
    print('Removed nonbinding BBV cuts: ', number_removed_constraints, ' Remaining: ', model._BBV_counter - number_removed_constraints)
#=======================================================================================================================
def remove_nonbinding_FCI(model):
    total_FCI_cuts = [constraint.name for constraint in model.constrs if constraint.name[:4] == 'FCI_']
    print('starting # of FCI + Gomory cuts: ', len(total_FCI_cuts))
    cuts_to_be_removed = [cut_number for cut_number in range(1, model._FCI_counter + 1) \
                          if abs(model.constr_by_name('FCI_' + str(cut_number)).expr.x) >= model._epsilon_3]
    number_removed_constraints = len(cuts_to_be_removed)
    model.remove([model.constr_by_name('FCI_' + str(cut_ID)) for cut_ID in cuts_to_be_removed])
    print('Removed nonbinding FCI cuts: ', number_removed_constraints, ' Remaining: ', model._FCI_counter - number_removed_constraints)
    Gomory_cuts_for_removed_FCIs = [constraint.name for constraint in model.constrs for cut_ID in cuts_to_be_removed \
                                    if constraint.name.startswith('FCI_' + str(cut_ID)+'_') \
                                    and abs(model.constr_by_name(constraint.name).expr.x) >= model._epsilon_3]
    model.remove([model.constr_by_name(name) for name in Gomory_cuts_for_removed_FCIs])
    print('Removed nonbinding Gomory cuts for nonbinding FCIs: ', len(Gomory_cuts_for_removed_FCIs))
    Gomory_FCI_cuts = [constraint.name for constraint in model.constrs if constraint.name[:4] == 'FCI_']
    cuts_to_be_removed = [cut_name for cut_name in Gomory_FCI_cuts \
                          if abs(model.constr_by_name(cut_name).expr.x) >= model._epsilon_3]
    model.remove([model.constr_by_name(code) for code in cuts_to_be_removed])
    print('Removed other nonbinding Gomory cuts for FCIs: ', len(cuts_to_be_removed))
    total_FCI_cuts = [constraint.name for constraint in model.constrs if constraint.name[:4] == 'FCI_']
    print('ending # of FCI + Gomory cuts: ', len(total_FCI_cuts))
# =======================================================================================================================
