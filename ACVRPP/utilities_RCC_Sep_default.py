from gurobipy import *
import numpy as np
from itertools import product
from time import *
#-----------------------------------------------------------------------------------------------------------------------
def create_RC_cut(model, S):
    fraction = model._n * (model._n + 1) / (2 * model._n - 2)
    target_no_vehicles = np.ceil(sum(model._demand[i] for i in S) / model._Q)
    S_size = len(S)
    if S_size <= fraction:
        Constr_LHS = quicksum(model._x[i, j] for (i, j) in product(S, S) if i != j)
        Constr_RHS = S_size - target_no_vehicles
    else:
        S_bar = model._C.difference(S)
        Constr_LHS = quicksum(model._x[i, j] for (i, j) in product(S_bar, S_bar) if i != j) + \
                     0.5 * quicksum(model._x[model._depot, i] + model._x[i, model._depot]for i in S_bar) - \
                     0.5 * quicksum(model._x[model._depot, i] + model._x[i, model._depot] for i in S)
        Constr_RHS = len(S_bar) - target_no_vehicles
    return Constr_LHS, Constr_RHS
#-----------------------------------------------------------------------------------------------------------------------
def separate_Rounded_Capacity_cuts(gurobi_model, opt_obj, verbose):
    epsilon_2 = pow(10, -4)
    epsilon_3 = pow(10, -3)
    iteration_counter = 0

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if GRB.Callback.MIPSOL_OBJBST >= epsilon_3:
                S = {i for i in model._C if model.cbGetSolution(delta[i]) > 0.5}
                Constr_LHS, Constr_RHS = create_RC_cut(gurobi_model, S)
                gurobi_model._cut_counter += 1
                #print('RCI_' + str(gurobi_model._cut_counter))
                gurobi_model.addLConstr(Constr_LHS, GRB.LESS_EQUAL, Constr_RHS, name='RCI_'+str(gurobi_model._cut_counter))

    if verbose:
        start_time = time()
    use_callback = True
    use_multiple_optimal_sol = False
    Q = gurobi_model._Q
    demand = gurobi_model._demand
    C = gurobi_model._C
    depot = gurobi_model._depot
    Edges_no_depot = {(i, j) for (i, j) in product(C, C) if i < j}
    m_separation = Model("Optimize")
    m_separation.setParam('OutputFlag', False)
    m_separation._C = C
    alpha = m_separation.addVar(vtype=GRB.INTEGER)
    delta = {}
    gamma = {}
    for (i, j) in Edges_no_depot:
        gamma[i, j] = m_separation.addVar(vtype=GRB.CONTINUOUS)
    for i in C:
        delta[i] = m_separation.addVar(vtype=GRB.BINARY)

    need_to_run_RCC_separation = True
    while need_to_run_RCC_separation:
        x_values = gurobi_model.getAttr("X", gurobi_model._x)
        m_separation.addLConstr(Q * alpha + 1, GRB.LESS_EQUAL, quicksum(delta[i] * demand[i] for i in C))
        m_separation.addLConstr(quicksum(delta[i] for i in C), GRB.GREATER_EQUAL, 3)

        relevant_edges = set()
        edge_capacities = {}
        for (i, j) in Edges_no_depot:
            edge_capacity = x_values[i, j] + x_values[j, i]
            if edge_capacity >= epsilon_2:
                relevant_edges.add((i, j))
                edge_capacities[i, j] = edge_capacity
                m_separation.addLConstr(gamma[i, j], GRB.LESS_EQUAL, delta[i])
                m_separation.addLConstr(gamma[i, j], GRB.LESS_EQUAL, delta[j])

        obj_separation = 2 + 2 * alpha - \
                         quicksum(delta[i] * (x_values[depot, i] + x_values[i, depot]) for i in C if x_values[depot, i] + x_values[i, depot] >= epsilon_2) - \
                         quicksum((delta[i] + delta[j] - 2 * gamma[i, j]) * edge_capacities[i, j] for (i, j) in relevant_edges)
        m_separation.setObjective(obj_separation, GRB.MAXIMIZE)

        if use_callback:
            m_separation.optimize(mycallback)
        else:
            m_separation.optimize()

        if m_separation.STATUS == 2:
            CVRP_model_size = gurobi_model.NumConstrs
            iteration_counter += 1
            if m_separation.objval >= epsilon_3:
                if use_multiple_optimal_sol:
                    for sol in range(m_separation.solcount):
                        m_separation.setParam('SolutionNumber', sol)
                        S = {i for i in C if delta[i].xn > 0.5}
                        Constr_LHS, Constr_RHS = create_RC_cut(gurobi_model, S)
                        gurobi_model._cut_counter += 1
                        #print('RCI_'+str(gurobi_model._cut_counter))
                        gurobi_model.addLConstr(Constr_LHS <= Constr_RHS, name='RCI_'+str(gurobi_model._cut_counter))
                else:
                    S = {i for i in C if delta[i].x > 0.5}
                    Constr_LHS, Constr_RHS = create_RC_cut(gurobi_model, S)
                    gurobi_model._cut_counter += 1
                    #print('RCI_' + str(gurobi_model._cut_counter))
                    gurobi_model.addLConstr(Constr_LHS <= Constr_RHS, name='RCI_'+str(gurobi_model._cut_counter))
                gurobi_model.optimize()
            if verbose:
                if iteration_counter % 100 == 0:
                    if opt_obj:
                        print(iteration_counter, ': ', round(time() - start_time, 1), ' sec. ', np.round(100 * gurobi_model.objval / opt_obj, 2),'%')
                    else:
                        print(iteration_counter, ': ', round(time() - start_time, 1), ' sec. ', np.round(gurobi_model.objval, 2))
            if gurobi_model.NumConstrs == CVRP_model_size:
                need_to_run_RCC_separation = False

        m_separation.remove(m_separation.getConstrs())
        m_separation.update()
    m_separation.dispose()
    return gurobi_model.objVal, iteration_counter
#-----------------------------------------------------------------------------------------------------------------------
def remove_nonbinding_RCI_cuts(model, cut_counter):
    RCI_cuts = [cut for cut in range(1, cut_counter + 1) if model.getConstrByName('RCI_' + str(cut))]
    removed_RCI_cuts = [cut for cut in RCI_cuts
                        if abs(model.getConstrByName('RCI_' + str(cut)).slack) >= 0.0001]
    model.remove([model.getConstrByName('RCI_' + str(cuts)) for cuts in removed_RCI_cuts])
    print('Removed nonbinding RCI cuts: ', len(removed_RCI_cuts))
    print('RCI cuts added: ', len(RCI_cuts) - len(removed_RCI_cuts))
#-----------------------------------------------------------------------------------------------------------------------
