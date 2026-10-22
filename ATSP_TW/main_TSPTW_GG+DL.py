#-----------------------------------------------------------------------------------------------------------------------
from gurobipy import *
import sys
import datetime
# appending a path
sys.path.insert(1, 'C:/Users/sa-kop/Documents/GG+DL_project/ATSP')
from TSP_utilities import *
from DFJ_cuts_utilities import *
from TW_preprocessing import *
#from random import *
import networkx as nx
from itertools import product
#-----------------------------------------------------------------------------------------------------------------------
def GG_ATSP_TW(path, filename, symmetry_flag, data_format_flag, n, GGE, GGEm, YCOS, YCOSm, lazy, preprocessing, NoRel, warm_start, relax, timelimit):
    V = set(range(n))
    d = load_instance(path, filename, n, symmetry_flag, data_format_flag)
    tmp = np.loadtxt(path + '/' + filename + '_tw.txt')
    a = np.zeros(n)
    b = np.zeros(n)
    for i in V:
        a[i] = tmp[i, 0]
        b[i] = tmp[i, 1]
    T = GRB.INFINITY
    if b[0] > 0:
        T = b[0]

    m = Model("Optimize")
    Arcs = [(i, j) for (i, j) in product(V, V) if i != j]
    Edges = [(i, j) for (i, j) in product(V, V) if i < j]
    Arcs_no_depot = [(i, j) for (i, j) in Arcs if i > 0 and j > 0]
    x = {}
    for (i, j) in Arcs:
        x[i, j] = m.addVar(vtype=GRB.CONTINUOUS, name='x_' + str(i) + '_' + str(j))
    for i in V:
        m.addLConstr(quicksum(x[i, j] for j in V - {i}) == 1)
        m.addLConstr(quicksum(x[j, i] for j in V - {i}) == 1)
    m._x = x
    G = nx.DiGraph()
    for (i, j) in Arcs:
        G.add_edge(i, j, weight=d[i, j])
    tau_tilde_from_depot = np.zeros(n)
    tau_tilde_to_depot = np.zeros(n)
    tau_tilde_from_depot_2 = np.zeros(n)
    tau_tilde_to_depot_2 = np.zeros(n)
    for i in V - {0}:
        tau_tilde_from_depot[i] = nx.shortest_path_length(G, source=0, target=i, weight="weight")
        tau_tilde_to_depot[i] = nx.shortest_path_length(G, source=i, target=0, weight="weight")
        G_tmp = G.copy()
        G_tmp.remove_edge(0, i)
        G_tmp.remove_edge(i, 0)
        tau_tilde_from_depot_2[i] = nx.shortest_path_length(G_tmp, source=0, target=i, weight="weight")
        tau_tilde_to_depot_2[i] = nx.shortest_path_length(G_tmp, source=i, target=0, weight="weight")
        #print(i, a[i], d[0, i], tau_tilde_from_depot[i], tau_tilde_from_depot_2[i])
        #print(i, a[i], tau_tilde_from_depot[i], tau_tilde_from_depot_2[i])

    tilde_a = a.copy()
    tilde_b = b.copy()
    for i in V - {0}:
        if a[i] <= tau_tilde_from_depot[i] - 0.0001:
            tilde_a[i] = tau_tilde_from_depot[i]
        if b[i] >= T - tau_tilde_to_depot[i] - 0.0001:
            tilde_b[i] = T - tau_tilde_to_depot[i]
    Eliminated_Arcs = [[i, 0] for i in V - {0} if tilde_a[i] + tau_tilde_to_depot[i] >= T + 0.001]
    if preprocessing:
        tilde_a, tilde_b, Eliminated_Arcs = update_TW_parameters(V, 0, d, tilde_a, tilde_b, Eliminated_Arcs, T, tau_tilde_from_depot, tau_tilde_to_depot)
        for i in V - {0}:
            if tilde_a[i] + tau_tilde_to_depot[i] >= T + 0.001:
                Eliminated_Arcs.append([i, 0])
    print('Total # tightened time bounds: ', \
        len({i for i in V - {0} if tilde_a[i] >=  0.0001 + a[i]}) + \
        len({i for i in V - {0} if tilde_a[i] <= -0.0001 + b[i]}), '# eliminated arcs: ', len(Eliminated_Arcs))

    for (i, j) in Eliminated_Arcs:
        m.addLConstr(x[i, j] == 0)

    if GGE and GGEm:
        GGE = False
    if YCOS and YCOSm:
        YCOS = False

    if (GGE or GGEm):
        g = {}
        for (i, j) in Arcs:
            if i > 0 and [i, j] not in Eliminated_Arcs:
                g[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
        inflow_distance = {}
        outflow_distance = {}
        if GGEm:
            for i in V - {0}:
                inflow_distance[i] = max(a[i], d[0, i]) * x[0, i] + \
                    quicksum(g[j, i] + (tilde_a[j] + d[j, i]) * x[j, i] for j in V - {0, i} if [j, i] not in Eliminated_Arcs)
                outflow_distance[i] = tilde_a[i] + quicksum(g[i, j] for j in V - {i} if [i, j] not in Eliminated_Arcs)
                m.addLConstr(inflow_distance[i] <= outflow_distance[i])
            for i in V - {0}:
                for j in V - {0, i}:
                    if [i, j] not in Eliminated_Arcs:
                        m_i_j = min(tilde_b[i], tilde_b[j] - d[i, j]) - tilde_a[i]
                        m.addLConstr(g[i, j] <= m_i_j * x[i, j])
                if [i, 0] not in Eliminated_Arcs:
                    if T < GRB.INFINITY:
                        m.addLConstr(g[i, 0] <= (min(tilde_b[i], (T - d[i, 0]) - tilde_a[i])) * x[i, 0])
        else:
            for i in V - {0}:
                inflow_distance[i] = max(a[i], d[0, i]) * x[0, i] + \
                    quicksum(g[j, i] + d[j, i] * x[j, i] for j in V - {0, i} if [j, i] not in Eliminated_Arcs)
                outflow_distance[i] = quicksum(g[i, j] for j in V - {i} if [i, j] not in Eliminated_Arcs)
                m.addLConstr(inflow_distance[i] <= outflow_distance[i])
            for i in V - {0}:
                for j in V - {i}:
                    if [i, j] not in Eliminated_Arcs:
                        m.addLConstr(g[i, j] >= tilde_a[i] * x[i, j])
                        m.addLConstr(g[i, j] <= tilde_b[i] * x[i, j])
            if T > 0 and T < GRB.INFINITY:
                m.addLConstr(quicksum(g[i, 0] + d[i, 0] * x[i, 0] for i in V - {0} if [i, 0] not in Eliminated_Arcs) <= T)

    if YCOSm or YCOS:
        time_label = {}
        for i in V - {0}:
            if YCOS:
                time_label[i] = m.addVar(lb=tilde_a[i], ub=tilde_b[i], vtype=GRB.CONTINUOUS)
                m.addLConstr(time_label[i] >= d[0, i] * x[0, i])
            if YCOSm:
                time_label[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
                # setting 1: full set of constraints:
                m.addLConstr(time_label[i] >= tilde_a[i] + \
                             quicksum(max(0, tilde_a[j] + d[j, i] - tilde_a[i]) * x[j, i] for j in V - {0, i} if [j, i] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] >= max(a[i], d[0, i]) * x[0, i] + \
                             quicksum(max(tilde_a[j] + d[j, i], tilde_a[i]) * x[j, i] for j in V - {0, i} if [j, i] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] >= max(a[i], d[0, i]) * x[0, i] + \
                             quicksum(max(max(a[j], d[0, j]) + d[j, i], tilde_a[i]) * x[0, j] for j in V - {0, i} if [0, j] not in Eliminated_Arcs))

                m.addLConstr(time_label[i] <= tilde_b[i] - \
                             quicksum(max(0, tilde_b[i] - tilde_b[j] + d[i, j]) * x[i, j] for j in V - {0, i} if [i, j] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] <= min(b[i], T - d[i, 0]) * x[i, 0] + \
                             quicksum(min(tilde_b[i], tilde_b[j] - d[i, j]) * x[i, j] for j in V - {0, i} if [i, j] not in Eliminated_Arcs))
                m.addLConstr(time_label[i] <= min(b[i], T - d[i, 0]) * x[i, 0] + \
                             quicksum(min(tilde_b[i], min(b[j], T - d[j, 0]) - d[i, j]) * x[j, 0] for j in V - {0, i} if [j, 0] not in Eliminated_Arcs))
                # end setting 1: full set of constraints:
            if T > 0 and (b[i] + tau_tilde_to_depot[i] > T) and [i, 0] not in Eliminated_Arcs:
                m.addLConstr(time_label[i] + d[i, 0] * x[i, 0] <= T)
        negative_B_j_i_count = 0
        for (i, j) in Arcs_no_depot:
            if [i, j] not in Eliminated_Arcs:
                M_i_j = tilde_b[i] - tilde_a[j] + d[i, j]
                if YCOSm:
                    B_j_i = tilde_b[i] - tilde_a[j] + min(-d[j, i], tilde_a[j] - tilde_a[i])
                    if B_j_i < 0:
                        negative_B_j_i_count += 1
                        B_j_i = 0
                else:
                    B_j_i = 0
                m.addLConstr(time_label[i] - time_label[j] + M_i_j * x[i, j] + B_j_i * x[j, i] <= M_i_j - d[i, j], name=str(i) + '_' + str(j))
        #print(negative_B_j_i_count)

    if (GGE or GGEm) and (YCOS or YCOSm):
        for i in V - {0}:
             m.addLConstr(outflow_distance[i] == time_label[i])

    obj = quicksum(d[i, j] * x[i, j] for (i, j) in x.keys())
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', False)

    LP_start_time = time()
    m.optimize()
    model_string = ''
    if GGE:
        model_string = 'GGE '
    if GGEm:
        model_string = 'GGE m. '
    if YCOS:
        model_string = model_string + 'YCOS '
    if YCOSm:
        model_string = model_string + 'YCOS m. '
    if lazy:
        model_string = model_string + 'lazy '
    if NoRel:
        model_string = model_string + 'NoRel'
    TSP_bound_0 = np.round(m.objVal, 2)
    print(model_string + ' TSPTW bound: ', np.round(m.objVal, 2), np.round(time() - LP_start_time, 1), ' sec.')
    add_cuts = True
    m._epsilon_1 = pow(10, -5)
    m._epsilon_2 = pow(10, -3)
    m._cuts = set()
    if add_cuts:
        new_bound, iteration_counter, execution_time = separate_DFJ_cuts(m, n)
        print('new TSPTW bound: ', new_bound)
        print('DFJ cuts identified, ', len(m._cuts), ' CPU:', round(execution_time, 1))
        remove_nonbinding_DFJ_cuts(m, m._cuts)
        TSP_bound_1 = np.round(m.objVal, 2)
        run_time = np.round(time() - LP_start_time, 1)
        print(' & $', round(TSP_bound_0, 2),'$', ' & $', round(TSP_bound_1, 2), '$', ' & $', str(run_time), '$')
    if not relax:
        #m.write(filename+'.lp')
        if warm_start:
            add_starting_solution(m, x, n, d, path, filename, True, symmetry_flag)
        for (i, j) in x.keys():
            x[i, j].vtype = GRB.BINARY

        #m.setParam('Cuts', 2)
        #m.setParam('Presolve', 0)
        #m.setParam('MIPFocus', 1)
        m.setParam('MIPGap', 0.0)
        #m.setParam('OutputFlag', True)
        #m.setParam('MIPGapAbs', 0.99)
        m.setParam('DisplayInterval', 100)
        if NoRel:
            m.setParam('NoRelHeurTime', timelimit * 0.3)
            m.setParam('NodeLimit', 0)
            m.optimize()
            print('Heuristic Norel Solution', m.objval, datetime.datetime.now())
        if (GGE or GGEm) and (YCOS or YCOSm) and lazy:
            m.update()
            for (i, j) in Arcs_no_depot:
                if [i, j] not in Eliminated_Arcs:
                    m.getConstrByName(str(i)+'_'+str(j)).Lazy = 3
        if NoRel:
            m.setParam('TimeLimit', timelimit * 0.7)
            m.setParam('NoRelHeurTime', 0)
            m.setParam('NodeLimit', GRB.INFINITY)
        else:
            m.setParam('TimeLimit', timelimit)

        m._Edges = Edges
        m._n = n
        def DFJ_callback(model, where):
            if where == GRB.Callback.MIPNODE:  # fractional node
                #if model.cbGet(GRB.callback.MIPNODE_NODCNT) % 1 == 0:  # frequency control
                if model.cbGet(GRB.callback.MIPNODE_NODCNT) <= 20000:    # frequency control
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:  # only search for violated cuts in feasible nodes
                        #x_values = model.cbGetNodeRel(model._x)
                        violated_DFJ_cuts = find_DFJ_cuts(model._n, model.cbGetNodeRel(model._x), model._epsilon_1, model._epsilon_2)
                        for S in violated_DFJ_cuts:
                            #S_code = set_to_code(S, n)
                            #print(S_code, S, S_code in m._cuts)
                            #m._cuts.add(S_code)
                            Constr_LHS = quicksum(model._x[i, j] for (i, j) in product(S, S) if i != j)
                            model.cbCut(Constr_LHS, GRB.LESS_EQUAL, len(S) - 1)
                            #if S_code not in m._cuts:
                            #    m._cuts.add(S_code)
                            #    Constr_LHS = quicksum(model._x[i, j] for (i, j) in product(S, S) if i != j)
                            #    model.cbCut(Constr_LHS, GRB.LESS_EQUAL, len(S) - 1)
        #m.optimize(DFJ_callback)
        m.optimize()
        #print(m.status, m.MIPGap)
        #if m.SolCount > 0:
        #    for (i, j) in x.keys():
        #        if x[i, j].x > 0.5:
        #            print(i, j, x[i, j].x)
        print(model_string, ' Objective ', round(m.objval, 2), ' Bound: ', round(m.objbound, 2), 'Gap: ', round(m.MIPGap * 100, 2))
        run_time = m.runtime
        if NoRel:
            run_time += timelimit * 0.3
        print('CPU:', round(run_time, 1))
        print(' & $', round(m.objval, 2),'$', ' & $', round(m.objbound, 2), '$', ' & $', str(round(m.MIPGap * 100, 2)), '$', ' & $', str(run_time), '$')
#-----------------------------------------------------------------------------------------------------------------------
def main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit):
    folder = 'SolomonPotvinBengio'
    warm_start = False
    # n = 20
    # filename = 'rc_201.1'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 26
    # filename = 'rc_201.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 32
    # filename = 'rc_201.3'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 26
    # filename = 'rc_201.4'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 33
    filename = 'rc_202.1'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 14
    # filename = 'rc_202.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 29
    # filename = 'rc_202.3'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 28
    # filename = 'rc_202.4'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 19
    # filename = 'rc_203.1'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 33
    # filename = 'rc_203.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 37
    filename = 'rc_203.3'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 15
    # filename = 'rc_203.4'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 46
    filename = 'rc_204.1'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 33
    filename = 'rc_204.2'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 24
    # filename = 'rc_204.3'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 14
    # filename = 'rc_205.1'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 27
    # filename = 'rc_205.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 35
    # filename = 'rc_205.3'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 28
    # filename = 'rc_205.4'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 37
    # filename = 'rc_206.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 25
    # filename = 'rc_206.3'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 38
    filename = 'rc_206.4'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 34
    # filename = 'rc_207.1'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 31
    filename = 'rc_207.2'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 33
    filename = 'rc_207.3'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 38
    filename = 'rc_208.1'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    # n = 29
    # filename = 'rc_208.2'
    # GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
    n = 36
    filename = 'rc_208.3'
    GG_ATSP_TW(folder, filename, False, 'SQUARE', n, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, warm_start, relax, timelimit)
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    timelimit = 3600
    relax = False
    lazy = False
    Preprocessing = True
    NoRel = True

    GGE = True
    GGEm = False
    YCOS = False
    YCOSm = False
    #main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)

    GGE = False
    GGEm = True
    YCOS = False
    YCOSm = False
    #main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)

    GGE = False
    GGEm = False
    YCOS = True
    YCOSm = False
    #main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)

    GGE = False
    GGEm = False
    YCOS = False
    YCOSm = True
    main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)

    GGE = False
    GGEm = True
    YCOS = False
    YCOSm = True
    main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)

    lazy = True
    main_ATSP_TW(relax, GGE, GGEm, YCOS, YCOSm, lazy, Preprocessing, NoRel, timelimit)