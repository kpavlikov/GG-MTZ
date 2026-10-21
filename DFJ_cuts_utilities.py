import numpy as np
from TSP_utilities import *
import networkx as nx
from itertools import product
from time import *
from gurobipy import quicksum
#import tsplib95
import os
#-----------------------------------------------------------------------------------------------------------------------
def find_DFJ_cuts(n, x_values, epsilon_1, epsilon_2):
    G = nx.Graph()
    cuts = []
    Edges = {(i, j) for (i, j) in x_values.keys() if i < j}
    #print(Edges)
    G_edges = [(u, v, {'capacity': x_values[u, v] + x_values[v, u]}) for (u, v) in Edges if x_values[u, v] + x_values[v, u] >= epsilon_1]
    G.add_edges_from(G_edges)
    # for (u, v) in model._Edges:
    #   edge_capacity = x_values[u, v] + x_values[v, u]
    #   if edge_capacity >= model._epsilon_1:
    #       G.add_edge(u, v, capacity=edge_capacity)
    T = nx.gomory_hu_tree(G)
    #print(T)
    # T_temp = T.copy()
    # for (i, j) in T_temp.edges:
    # cut_value = T[i][j]['weight']
    # if cut_value <= 2 - model._epsilon_2:
    for (i, j) in {(i, j) for (i, j) in T.edges if T[i][j]['weight'] <= 2 - epsilon_2}:
        T.remove_edge(*(i, j))
        U, V = tuple(nx.connected_components(T))
        #print(len(V) + len(U))
        T.add_edge(i, j, weight=1)
        # T.add_edge(i, j, weight=T_temp[i][j]['weight'])
        if len(U) <= n / 2:
            S = U
        else:
            S = V
        if len(S) > 1:
            cuts.append(S)
    return cuts
#-----------------------------------------------------------------------------------------------------------------------
def separate_DFJ_cuts(gurobi_model, n):
    need_to_run_DFJ_separation = True
    iteration_counter = 0
    start_time = time()
    while need_to_run_DFJ_separation:
        iteration_counter += 1
        x_values = {(i, j): gurobi_model._x[i, j].x for (i, j) in gurobi_model._x.keys()}
        #print(x_values)
        violated_DFJ_cuts = find_DFJ_cuts(n, x_values, gurobi_model._epsilon_1, gurobi_model._epsilon_2)
        #print(len(violated_DFJ_cuts))
        Found_new_cuts = False
        for S in violated_DFJ_cuts:
            S_code = set_to_code(S, n)
            Found_new_cuts = True
            gurobi_model._cuts.add(S_code)
            Constr_LHS = quicksum(gurobi_model._x[i, j] for (i, j) in product(S, S) if i != j)
            gurobi_model.addLConstr(Constr_LHS <= len(S) - 1, name=S_code)
            #print(S_code, S)
            #if S_code not in gurobi_model._cuts:
            #    Found_new_cuts = True
            #    gurobi_model._cuts.add(S_code)
            #    Constr_LHS = quicksum(gurobi_model._x[i, j] for (i, j) in product(S, S) if i != j)
            #    gurobi_model.addLConstr(Constr_LHS <= len(S) - 1, name=S_code)
        if Found_new_cuts:
            gurobi_model.optimize()
            print(round(gurobi_model.objval, 2))
        else:
            need_to_run_DFJ_separation = False
            execution_time = round(time() - start_time, 1)
    return np.round(gurobi_model.objVal, 2), iteration_counter, execution_time
#-----------------------------------------------------------------------------------------------------------------------
def remove_nonbinding_DFJ_cuts(model, cuts):
    removed_DFJ_cuts = [cut for cut in cuts
                            if abs(model.getConstrByName(cut).slack) >= 0.0001]
    model.remove([model.getConstrByName(cut) for cut in removed_DFJ_cuts])
    print('Removed nonbinding DFJ cuts: ', len(removed_DFJ_cuts))
    print('DFJ cuts added: ', len(cuts) - len(removed_DFJ_cuts))
#-----------------------------------------------------------------------------------------------------------------------
