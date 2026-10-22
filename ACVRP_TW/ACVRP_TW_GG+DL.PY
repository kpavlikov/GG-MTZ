#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import networkx as nx
from itertools import product
from gurobipy import *
from time import *
import sys
path = os.path.dirname( __file__ )
sys.path.append(os.path.dirname(path) + '/ACVRP')
from utilities_RCC_Sep_default import *
sys.path.append(os.path.dirname(path) + '/ATSP_TW')
from TW_preprocessing import *
#-----------------------------------------------------------------------------------------------------------------------
def adjust_capacity(demand, Q):
    m = Model("Optimize")
    m.setParam('OutputFlag', False)
    n = len(demand)
    x = {}
    for i in range(0, n):
        x[i] = m.addVar(vtype=GRB.BINARY)
    m.addLConstr(Q >= quicksum(x[i] * demand[i] for i in range(0, n)))
    m.setObjective(quicksum(x[i] * demand[i] for i in range(0, n)), GRB.MAXIMIZE)
    m.optimize()
    return m.objval
#-----------------------------------------------------------------------------------------------------------------------
def CVRP_TW_DL(folder, filename, opt_obj, n, Q, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, warm_start, timelim):
    C_0 = set(range(0, n + 1))
    depot = 0
    C = C_0 - {depot}
    temp = np.loadtxt(folder + '/' + filename + '.txt')
    c = np.zeros((n + 1, n + 1))
    Tau = np.zeros((n + 1, n + 1))
    demand = np.zeros(n + 1)
    a = np.zeros(n + 1)
    b = np.zeros(n + 1)
    service_time = np.zeros(n + 1)
    Arcs = [(i, j) for (i, j) in product(C_0, C_0) if i != j]
    Arcs_no_depot = [(i, j) for (i, j) in product(C, C) if i != j]
    for i in C_0:
        demand[i] = temp[i, 3]
        a[i] = temp[i, 4]
        b[i] = temp[i, 5]
        service_time[i] = temp[i, 6]
        for j in C_0:
            if i < j:
                #c[i, j] = np.round(np.sqrt(np.square(temp[i, 1] - temp[j, 1]) + np.square(temp[i, 2] - temp[j, 2])), 2)
                c[i, j] = 0.1 * np.floor(10 * np.sqrt(np.square(temp[i, 1] - temp[j, 1]) + np.square(temp[i, 2] - temp[j, 2])))
                c[j, i] = c[i, j]
                Tau[i, j] = service_time[i] + c[i, j]
                Tau[j, i] = Tau[i, j]
    G = nx.DiGraph()
    for i in C:
        G.add_edge(depot, i, weight=Tau[depot, i])
        G.add_edge(i, depot, weight=Tau[i, depot])
    for (i, j) in Arcs_no_depot:
        G.add_edge(i, j, weight=Tau[i, j])
    tau_tilde_from_depot = np.zeros(n + 1)
    tau_tilde_to_depot = np.zeros(n + 1)
    for i in C:
        tau_tilde_from_depot[i] = nx.shortest_path_length(G, source=depot, target=i, weight="weight")
        tau_tilde_to_depot[i] = nx.shortest_path_length(G, source=i, target=depot, weight="weight")

    print('Min. # of vehicles: ', np.ceil(sum(demand) / Q))
    #new_Q = adjust_capacity(demand, Q)
    #print(Q, new_Q)
    if b[depot] > 0:
        T = b[depot]
    else:
        T = GRB.INFINITY
    m = Model("Optimize")
    x = {}
    for (i, j) in Arcs:
        x[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
    for i in C:
        m.addLConstr(quicksum(x[i, j] for j in C_0 - {i}) == 1)
        m.addLConstr(quicksum(x[j, i] for j in C_0 - {i}) == 1)

    obj = quicksum(c[i, j] * x[i, j] for (i, j) in Arcs)
    m.setObjective(obj, GRB.MINIMIZE)

    Eliminated_Arcs = [[i, j] for (i, j) in Arcs_no_depot if Q < demand[i] + demand[j]]
    #print(Eliminated_Arcs)
    tilde_a = a.copy()
    tilde_b = b.copy()
    for i in C:
        if a[i] <= tau_tilde_from_depot[i] - 0.0001:
            tilde_a[i] = tau_tilde_from_depot[i]
        if b[i] >= T - tau_tilde_to_depot[i] - 0.0001:
            tilde_b[i] = T - tau_tilde_to_depot[i]
    for i in C:
        if tilde_a[i] + tau_tilde_to_depot[i] >= T + 0.001:
            Eliminated_Arcs.append([i, depot])
    if Preprocessing:
        tilde_a, tilde_b, Eliminated_Arcs = update_TW_parameters(C_0, 0, Tau, tilde_a, tilde_b, Eliminated_Arcs, T, tau_tilde_from_depot, tau_tilde_to_depot)
        for i in C:
            if tilde_a[i] + tau_tilde_to_depot[i] >= T + 0.001:
                Eliminated_Arcs.append([i, 0])
            #print(i, a[i], tilde_a[i], tilde_b[i], b[i])
    print('Total # tightened time bounds: ', \
        len({i for i in C if tilde_a[i] >=  0.0001 + a[i]}) + \
        len({i for i in C if tilde_a[i] <= -0.0001 + b[i]}), '# eliminated arcs: ', len(Eliminated_Arcs))

    for (i, j) in Eliminated_Arcs:
            m.addLConstr(x[i, j] == 0)

    GG_capacity = True
    # Gavish and Graves modelling for capacity constraints
    if GG_capacity:
        f = {}
        for [i, j] in Arcs_no_depot:
            if [i, j] not in Eliminated_Arcs:
                f[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
        for i in C:
            if [i, depot] not in Eliminated_Arcs:
                f[i, depot] = m.addVar(vtype=GRB.CONTINUOUS)
        inflow = {}
        outflow = {}
        modified = True
        for i in C:
            if modified:
                inflow[i] = Q * x[depot, i] + quicksum(f[j, i] + demand[i] * x[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs)
                if [i, depot] not in Eliminated_Arcs:
                    outflow[i] = quicksum(f[i, j] + demand[j] * x[i, j] for j in C - {i} if [i, j] not in Eliminated_Arcs) + f[i, depot]
                else:
                    outflow[i] = quicksum(f[i, j] + demand[j] * x[i, j] for j in C - {i} if [i, j] not in Eliminated_Arcs)
                for j in C - {i}:
                    if [i, j] not in Eliminated_Arcs:
                        m.addLConstr(f[i, j] <= (Q - demand[i] - demand[j]) * x[i, j], name='flow_'+str(i)+'_'+str(j))
            else:
                inflow[i] = quicksum(f[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs)
                outflow[i] = quicksum(f[i, j] for j in C - {i} if [i, j] not in Eliminated_Arcs)
                for j in C - {i}:
                    if [i, j] not in Eliminated_Arcs:
                        m.addLConstr(f[i, j] <= Q * x[i, j])
            m.addLConstr(inflow[i] == outflow[i] + demand[i])
    # MTZ modelling for capacity constraints
    DL_capacity = True
    if DL_capacity:
        u = {}
        for i in C:
            u[i] = m.addVar(vtype=GRB.CONTINUOUS, name='u_' + str(i))
        for i in C:
            adjcoeff = max(demand[j] for j in C - {i})
            m.addLConstr(u[i] <= Q - (Q - adjcoeff - demand[i]) * x[depot, i] - quicksum(demand[j] * x[i, j] for j in C - {i}))
            m.addLConstr(u[i] >= demand[i] + quicksum(demand[j] * x[j, i] for j in C - {i}))
            for j in C - {i}:
                if [i, j] not in Eliminated_Arcs:
                    m.addLConstr(u[i] - u[j] + Q * x[i, j] + (Q - demand[i] - demand[j]) * x[j, i] <= Q - demand[j], name='DL_capacity_'+str(i)+'_'+str(j))
    if GG_capacity and DL_capacity:
        for i in C:
            m.addLConstr(outflow[i] == Q - u[i])

    # "Gavish and Graves" modelling for time windows constraints
    if GGEm or GGE:
        g = {}
        for (i, j) in Arcs_no_depot:
            if [i, j] not in Eliminated_Arcs:
                g[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
        for i in C:
            if [i, depot] not in Eliminated_Arcs:
                g[i, depot] = m.addVar(vtype=GRB.CONTINUOUS)
        inflow_time = {}
        outflow_time = {}
        if GGEm:
            for i in C:
                inflow_time[i] = max(a[i], c[depot, i]) * x[depot, i] + \
                                 quicksum(g[j, i] + (tilde_a[j] + service_time[j] + c[j, i]) * x[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs)
                outflow_time[i] = tilde_a[i] + quicksum(g[i, j] for j in C_0 - {i} if [i, j] not in Eliminated_Arcs)
                m.addLConstr(inflow_time[i] <= outflow_time[i])
            for i in C:
                for j in C - {i}:
                    if [i, j] not in Eliminated_Arcs:
                        M_ij = min(tilde_b[i], tilde_b[j] - service_time[i] - c[i, j]) - tilde_a[i]
                        m.addLConstr(g[i, j] <= M_ij * x[i, j])
                if [i, depot] not in Eliminated_Arcs:
                    if T > 0 and T < GRB.INFINITY:
                        m.addLConstr(g[i, depot] <= (min(tilde_b[i], T - service_time[i] - c[i, depot]) - tilde_a[i]) * x[i, depot])
                    else:
                        m.addLConstr(g[i, depot] <= (tilde_b[i] - tilde_a[i]) * x[i, depot])
        else:
            inflow_time[i] = max(a[i], c[depot, i]) * x[depot, i] + \
                             quicksum(g[j, i] + (service_time[j] + c[j, i]) * x[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs)
            outflow_time[i] = quicksum(g[i, j] for j in C_0 - {i} if [i, j] not in Eliminated_Arcs)
            m.addLConstr(inflow_time[i] <= outflow_time[i])
            for i in C:
                for j in C_0 - {i}:
                    if [i, j] not in Eliminated_Arcs:
                        m.addLConstr(g[i, j] >= tilde_a[i] * x[i, j])
                        m.addLConstr(g[i, j] <= tilde_b[i] * x[i, j])
                if [i, depot] not in Eliminated_Arcs:
                    if T > 0 and T < GRB.INFINITY:
                        m.addLConstr(g[i, depot] + service_time[i] + c[i, depot] * x[i, depot] <= T)

    # "MTZ" modelling for time windows constraints
    if YCOS or YCOSm:
        time_label = {}
        if YCOSm:
            for i in C:
                time_label[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
                m.addLConstr(time_label[i] >= tilde_a[i] + \
                             quicksum(max(0, tilde_a[j] + service_time[j] + c[j, i] - tilde_a[i]) * x[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] >= max(a[i], c[depot, i]) * x[depot, i] + \
                             quicksum(max(tilde_a[i], tilde_a[j] + service_time[j] + c[j, i]) * x[j, i] for j in C - {i} if [j, i] not in Eliminated_Arcs))

                m.addLConstr(time_label[i] <= tilde_b[i] - \
                             quicksum(max(0, tilde_b[i] - tilde_b[j] + service_time[i] + c[i, j]) * x[i, j] for j in C - {i} if [i, j] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] <= min(b[i], T - service_time[i] - c[i, depot]) * x[i, depot] + \
                             quicksum(min(tilde_b[i], tilde_b[j] - service_time[i] - c[i, j]) * x[i, j] for j in C - {i} if [i, j] not in Eliminated_Arcs))
            for (i, j) in Arcs_no_depot:
                if [i, j] not in Eliminated_Arcs:
                    M_i_j = tilde_b[i] - tilde_a[j] + c[i, j] + service_time[i]
                    B_j_i = M_i_j - c[i, j] - service_time[i] + min(-c[j, i] - service_time[j], tilde_a[j] - tilde_a[i])
                    if B_j_i > 0:
                        m.addLConstr(time_label[i] - time_label[j] + M_i_j * x[i, j] + B_j_i * x[j, i] <= M_i_j - c[i, j] - service_time[i], name='DL_time_'+str(i)+'_'+str(j))
                    else:
                        m.addLConstr(time_label[i] - time_label[j] + M_i_j * x[i, j] <= M_i_j - c[i, j] - service_time[i], name='DL_time_'+str(i)+'_'+str(j))
        else:
            for i in C:
                time_label[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
                m.addLConstr(time_label[i] >= c[depot, i] * x[depot, i])
                m.addLConstr(time_label[i] >= tilde_a[i])
                m.addLConstr(time_label[i] <= tilde_b[i])
            for (i, j) in Arcs_no_depot:
                if [i, j] not in Eliminated_Arcs:
                    M_i_j = tilde_b[i] - tilde_a[j] + c[i, j] + service_time[i]
                    B_j_i = M_i_j - c[i, j] - service_time[i] + min(-c[j, i] - service_time[j], tilde_a[j] - tilde_a[i])
                    m.addLConstr(time_label[i] - time_label[j] + M_i_j * x[i, j] + B_j_i * x[j, i] <= M_i_j - c[i, j] - service_time[i], name='DL_time_' + str(i) + '_' + str(j))

    if (GGE or GGEm) and (YCOS or YCOSm):
        for i in C:
            m.addLConstr(outflow_time[i] == time_label[i])

    m.setParam('OutputFlag', False)
    start_time = time()
    m.optimize()
    run_time = round(m.Runtime, 1)
    cvrp_lower_bound_0 = m.objval
    print('Instance ', filename, 'Q =', str(Q))
    model_string = 'TW settings: '
    if GGE:
        model_string = model_string + ' GGE'
    if GGEm:
        model_string = model_string + ' GGE m.'
    if YCOS:
        model_string = model_string + ' YCOS'
    if YCOSm:
        model_string = model_string + ' YCOS m.'
    if lazy:
        model_string = model_string + ' lazy'
    if opt_obj:
        fraction = np.round(100 * cvrp_lower_bound_0 / opt_obj, 2)
        print(model_string + ' ACVRP-TW lower bound ', round(cvrp_lower_bound_0, 2), ' ', fraction, '% ', str(run_time)+' sec.')
    else:
        print(model_string + ' ACVRP-TW lower bound ', round(cvrp_lower_bound_0, 2), ' ', str(run_time) + ' sec.')

    if run_RCC_sep:
        m._n = n
        m._Q = Q
        m._demand = demand
        m._x = x
        m._depot = depot
        m._C = C
        m._cut_counter = 0
        rcc_separation_start_time = time()
        cvrp_lower_bound, iteration_count = separate_Rounded_Capacity_cuts(m, None, verbose=True)
        run_time = round(time() - rcc_separation_start_time, 1)
        print('RCC separation: ', 'cuts =', str(m._cut_counter), ' iterations =', str(iteration_count))
        if opt_obj:
            fraction = np.round(100 * cvrp_lower_bound / opt_obj, 2)
            print('Final CVRP lower bound ', round(cvrp_lower_bound, 2), ' ', fraction, '% ', str(run_time)+' sec.')
        else:
            print('Final CVRP lower bound ', round(cvrp_lower_bound, 2), ' ', str(run_time) + ' sec.')
        print(' & $', round(cvrp_lower_bound_0, 2),'$', ' & $', round(cvrp_lower_bound, 2), '$', ' & $', str(run_time), '$')
    if not relax:
        if warm_start:
            solution = np.loadtxt(folder + '/' + filename + '_solution.txt')
            solution_objective = 0
            for k in range(1, len(solution) - 1):
                if int(solution[k]) == 0:
                    x[0, int(solution[k + 1])].start = 1
                    solution_objective += c[0, int(solution[k + 1])]
                elif int(solution[k + 1]) == 0:
                    x[int(solution[k]), 0].start = 1
                    solution_objective += c[int(solution[k]), 0]
                else:
                    solution_objective += c[int(solution[k]), int(solution[k + 1])]
                    x[int(solution[k]), int(solution[k + 1])].start = 1

        for (i, j) in x.keys():
            x[i, j].vtype = GRB.BINARY
        m.setParam('OutputFlag', True)
        start_time = time()
        m.setParam('MIPGap', 0.0)
        m.setParam('TimeLimit', timelim)
        m.setParam('NoRelHeurTime', timelim * 0.7)
        #m.setParam('Cuts', 3)
        m.setParam('DisplayInterval', 100)
        if (GGE or GGEm) and (YCOS or YCOSm) and lazy:
            m.update()
            for (i, j) in Arcs_no_depot:
                if [i, j] not in Eliminated_Arcs:
                    m.getConstrByName('DL_capacity_'+str(i)+'_'+str(j)).Lazy = 3
                    m.getConstrByName('DL_time_'+str(i)+'_'+str(j)).Lazy = 3
        m.optimize()
        print('ACVRP TW solution ', str(round(m.objval, 2)), str(round(m.objbound, 2)), ' ', str(round(m.MIPGap * 100, 2)), '% ', str(round(time() - start_time, 1)) + ' sec.')
    m.dispose()
    print()
#-----------------------------------------------------------------------------------------------------------------------
def main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim):
    run_RCC_sep = True
    warm_start = False
    CVRP_TW_DL('Solomon', 'C101', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C102', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C103', 826.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C104', 822.9, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C105', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C106', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C107', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C108', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C109', 827.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C201', 589.1, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C202', 589.1, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C203', 588.7, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C204', 588.1, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C205', 586.4, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C206', 586.0, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C207', 585.8, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'C208', 585.8, 100, 700, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R101', 1637.7, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R102', 1466.6, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R103', 1208.7, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R104', 971.5, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R105', 1355.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R106', 1234.6, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R107', 1064.6, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R108', 932.1, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R109', 1146.9, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R110', 1068.0, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R111', 1048.7, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R112', 948.6, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R201', 1143.2, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R202', 1029.6, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R203', 870.8, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R204', 731.3, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R205', 949.8, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R206', 875.9, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R207', 794.0, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R208', 701.0, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R209', 854.8, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R210', 900.5, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'R211', 746.7, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC101', 1619.8, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC102', 1457.4, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC103', 1258.0, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC104', 1132.3, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC105', 1513.7, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC106', 1372.7, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC107', 1207.8, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC108', 1114.2, 100, 200, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC201', 1261.8, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC202', 1092.3, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC203', 923.7, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC204', 783.5, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC205', 1154.0, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC206', 1051.1, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC207', 962.9, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
    CVRP_TW_DL('Solomon', 'RC208', 776.1, 100, 1000, relax, run_RCC_sep, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, timelim)
#-----------------------------------------------------------------------------------------------------------------------
def main_Solomon_CVRP_TW_total():
    timelim = 300
    relax = False
    Preprocessing = True
    lazy = False
    NoRel = True

    GGE = True
    YCOS = False
    GGEm = False
    YCOSm = False
    #main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)

    GGE = False
    YCOS = False
    GGEm = True
    YCOSm = False
    main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)

    GGE = False
    YCOS = True
    GGEm = False
    YCOSm = False
    #main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)

    GGE = False
    YCOS = False
    GGEm = False
    YCOSm = True
    main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)

    GGE = False
    YCOS = False
    GGEm = True
    YCOSm = True
    main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)

    lazy = True
    #main_Solomon_CVRP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelim)
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main_Solomon_CVRP_TW_total()