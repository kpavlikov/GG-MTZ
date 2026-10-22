#-----------------------------------------------------------------------------------------------------------------------
#import numpy as np
#from gurobipy import *
from utilities_RCC_Sep_default import *
#-----------------------------------------------------------------------------------------------------------------------
def GG_DL(folder, filename, GG, modified, DL, lazy, n, Q, depot, relax, timelim):
    d = np.loadtxt(folder + '/' + filename + '.dat')
    temp = np.loadtxt(folder + '/' + filename + '+capacities.dat')
    demand = temp[:, 1]

    m = Model("Optimize")
    x = {}
    C_0 = set()
    for i in range(0, n):
        C_0.add(int(temp[i, 0]))
    C = C_0 - {depot}

    Arcs = {(i, j) for (i, j) in product(C_0, C_0) if i != j}
    Arcs_no_depot = {(i, j) for (i, j) in product(C, C) if i != j}
    for (i, j) in Arcs:
        x[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
    for i in C:
        m.addLConstr(quicksum(x[i, j] for j in C_0 - {i}), GRB.EQUAL, 1)
        m.addLConstr(quicksum(x[j, i] for j in C_0 - {i}), GRB.EQUAL, 1)

    obj = quicksum(d[i, j] * x[i, j] for (i, j) in Arcs)
    m.setObjective(obj, GRB.MINIMIZE)

    if DL:
        u = {}
        for i in C:
            u[i] = m.addVar(vtype=GRB.CONTINUOUS)
        for i in C:
            adjcoeff = max(demand[j] for j in C - {i})
            m.addLConstr(u[i], GRB.LESS_EQUAL, Q - (Q - adjcoeff - demand[i]) * x[depot, i] - quicksum(demand[j] * x[i, j] for j in C - {i}))
            m.addLConstr(u[i], GRB.GREATER_EQUAL, demand[i] + quicksum(demand[j] * x[j, i] for j in C - {i}))
        for (i, j) in Arcs_no_depot:
            m.addLConstr(u[i] - u[j] + Q * x[i, j] + (Q - demand[i] - demand[j]) * x[j, i], GRB.LESS_EQUAL, Q - demand[j], name='DL_'+str(i)+'_'+str(j))
    if GG:
        y = {}
        inflow = {}
        outflow = {}
        for (i, j) in Arcs_no_depot:
            y[i, j] = m.addVar(vtype=GRB.CONTINUOUS)
        for i in C:
            y[i, depot] = m.addVar(vtype=GRB.CONTINUOUS)
            if modified:
                inflow[i] = Q * x[depot, i] + quicksum(y[j, i] + demand[i] * x[j, i] for j in C - {i})
                outflow[i] = y[i, depot] + quicksum(y[i, j] + demand[j] * x[i, j] for j in C - {i})
            else:
                inflow[i] = Q * x[depot, i] + quicksum(y[j, i] for j in C - {i})
                outflow[i] = y[i, depot] + quicksum(y[i, j] for j in C - {i})
            m.addLConstr(inflow[i] == outflow[i] + demand[i])
        if modified:
            for (i, j) in Arcs_no_depot:
                m.addLConstr(y[i, j], GRB.LESS_EQUAL, (Q - demand[i] - demand[j]) * x[i, j], name='flow_'+str(i)+'_'+str(j))
        else:
            for (i, j) in Arcs_no_depot:
                m.addLConstr(y[i, j], GRB.LESS_EQUAL, Q * x[i, j])
    if GG and DL:
        for i in C:
            m.addLConstr(outflow[i] == Q - u[i])

    m.setParam('OutputFlag', False)
    m.optimize()
    run_time = round(m.Runtime, 1)
    cvrp_lower_bound = m.objval
    print('Instance: ', filename, 'Q =', str(Q))
    print('CVRP lower bound: ', round(cvrp_lower_bound, 2), ' ', str(run_time) + ' sec.')
    run_RCC_sep = True
    if run_RCC_sep:
        m._C = C
        m._x = x
        m._n = n
        m._depot = depot
        m._demand = demand
        m._Q = Q
        m._cut_counter = 0
        rcc_separation_start_time = time()
        cvrp_lower_bound, iteration_count = separate_Rounded_Capacity_cuts(m, None, verbose=False)
        run_time = round(time() - rcc_separation_start_time, 1)
        print('RCC separation: ', 'cuts =', str(m._cut_counter), ' iterations =', str(iteration_count))
        remove_nonbinding_RCI_cuts(m, m._cut_counter)
        print('GG: ', str(GG), 'GGm: ', str(modified), 'DL: ', str(DL), 'lazy: ', lazy)
        print('Final CVRP lower bound: ', round(cvrp_lower_bound, 2), ' ', str(run_time) + ' sec.')
        print()
    if not relax:
        for (i, j) in x.keys():
            x[i, j].vtype = GRB.BINARY
        m.update()
        if GG and DL:
            if lazy:
                for (i, j) in Arcs_no_depot:
                    m.getConstrByName('DL_'+str(i)+'_'+str(j)).Lazy = 3
        #m.setParam('Presolve', 0)
        #m.setParam('OutputFlag', True)
        #m.setParam('DisplayInterval', 100)
        m.setParam('MIPGap', 0)
        m.setParam('MIPGapAbs', 0.99)
        #m.setParam('MIPFocus', 2)
        #m.setParam('Cuts', 2)
        #m.write(filename+'aggregate.lp')
        m.setParam('TimeLimit', timelim)
        m.optimize()
        print(filename, 'Q=' + str(Q), round(m.objval, 2), round(m.objbound, 2), str(round(m.MIPGap * 100, 2)) + '%', str(round(m.Runtime, 1)) + ' sec.')
        print()
        m.dispose()
#-----------------------------------------------------------------------------------------------------------------------
def main_ACVRP(relax, GG, m, DL, lazy):
    timelim = 3600
    #GG_DL('ACVRP', 'A034-02f', GG, m, DL, lazy, 34, 1000, 33, relax, timelim)
    # GG_DL('ACVRP', 'A034-02f', GG, m, DL, lazy, 34, 700, 33, relax, timelim)
    GG_DL('ACVRP', 'A034-02f', GG, m, DL, lazy, 34, 500, 33, relax, timelim)
    # GG_DL('ACVRP', 'A034-02f', GG, m, DL, lazy, 34, 400, 33, relax, timelim)
    # GG_DL('ACVRP', 'A036-03f', GG, m, DL, lazy, 36, 1000, 35, relax, timelim)
    # GG_DL('ACVRP', 'A036-03f', GG, m, DL, lazy, 36, 750, 35, relax, timelim)
    # GG_DL('ACVRP', 'A036-03f', GG, m, DL, lazy, 36, 600, 35, relax, timelim)
    # GG_DL('ACVRP', 'A036-03f', GG, m, DL, lazy, 36, 500, 35, relax, timelim)
    # GG_DL('ACVRP', 'A039-03f', GG, m, DL, lazy, 39, 1000, 38, relax, timelim)
    # GG_DL('ACVRP', 'A039-03f', GG, m, DL, lazy, 39, 750, 38, relax, timelim)
    # GG_DL('ACVRP', 'A039-03f', GG, m, DL, lazy, 39, 600, 38, relax, timelim)
    # GG_DL('ACVRP', 'A039-03f', GG, m, DL, lazy, 39, 500, 38, relax, timelim)
    # GG_DL('ACVRP', 'A045-03f', GG, m, DL, lazy, 45, 1000, 44, relax, timelim)
    # GG_DL('ACVRP', 'A045-03f', GG, m, DL, lazy, 45, 750, 44, relax, timelim)
    # GG_DL('ACVRP', 'A045-03f', GG, m, DL, lazy, 45, 600, 44, relax, timelim)
    # GG_DL('ACVRP', 'A045-03f', GG, m, DL, lazy, 45, 500, 44, relax, timelim)
    # GG_DL('ACVRP', 'A048-03f', GG, m, DL, lazy, 48, 1000, 47, relax, timelim)
    # GG_DL('ACVRP', 'A048-03f', GG, m, DL, lazy, 48, 750, 47, relax, timelim)
    # GG_DL('ACVRP', 'A048-03f', GG, m, DL, lazy, 48, 600, 47, relax, timelim)
    # GG_DL('ACVRP', 'A048-03f', GG, m, DL, lazy, 48, 500, 47, relax, timelim)
    # GG_DL('ACVRP', 'A056-03f', GG, m, DL, lazy, 56, 1000, 55, relax, timelim)
    # GG_DL('ACVRP', 'A056-03f', GG, m, DL, lazy, 56, 750, 55, relax, timelim)
    # GG_DL('ACVRP', 'A056-03f', GG, m, DL, lazy, 56, 600, 55, relax, timelim)
    # GG_DL('ACVRP', 'A056-03f', GG, m, DL, lazy, 56, 500, 55, relax, timelim)
    # GG_DL('ACVRP', 'A065-03f', GG, m, DL, lazy, 65, 1000, 64, relax, timelim)
    # GG_DL('ACVRP', 'A065-03f', GG, m, DL, lazy, 65, 750, 64, relax, timelim)
    # GG_DL('ACVRP', 'A065-03f', GG, m, DL, lazy, 65, 600, 64, relax, timelim)
    # GG_DL('ACVRP', 'A065-03f', GG, m, DL, lazy, 65, 500, 64, relax, timelim)
    # GG_DL('ACVRP', 'A071-03f', GG, m, DL, lazy, 71, 1000, 70, relax, timelim)
    # GG_DL('ACVRP', 'A071-03f', GG, m, DL, lazy, 71, 750, 70, relax, timelim)
    # GG_DL('ACVRP', 'A071-03f', GG, m, DL, lazy, 71, 600, 70, relax, timelim)
    # GG_DL('ACVRP', 'A071-03f', GG, m, DL, lazy, 71, 500, 70, relax, timelim)
# ========================================================================================================
if __name__ == "__main__":
    relax = False

    #GG = True
    #m = False
    #DL = False
    #lazy = False
    #main_ACVRP(relax, GG, m, DL, lazy)

    GG = True
    m = True
    DL = False
    lazy = False
    main_ACVRP(relax, GG, m, DL, lazy)

    GG = False
    m = False
    DL = True
    lazy = False
    #main_ACVRP(relax, GG, m, DL, lazy)

    GG = True
    m = True
    DL = True
    lazy = False
    #main_ACVRP(relax, GG, m, DL, lazy)

    GG = True
    m = True
    DL = True
    lazy = True
    #main_ACVRP(relax, GG, m, DL, lazy)