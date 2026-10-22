from time import *
from gurobipy import *
from TSP_utilities import *
from DFJ_cuts_utilities import *
from itertools import product
#========================================================================================================
def GG_DL_ATSP(folder, filename, n, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit):
    V = set(range(n))
    Arcs_no_depot = [(i, j) for (i, j) in product(V - {0}, V - {0}) if i != j]
    Arcs = [(i, j) for (i, j) in product(V, V) if i != j]
    d = load_instance(folder, filename, n, False, 'VECTOR')

    m = Model()
    m.setParam('OutputFlag', False)

    # variables indicating if arc (i, j) is used in the route or not
    x = {}
    for (i, j) in Arcs:
        x[i, j] = m.addVar(vtype=GRB.CONTINUOUS, name='x_' + str(i) + '_' + str(j))
    for i in V:
        m.addLConstr(quicksum(x[i, j] for j in V - {i}), GRB.EQUAL, 1)
        m.addLConstr(quicksum(x[j, i] for j in V - {i}), GRB.EQUAL, 1)
    m._x = x
    m._epsilon_1 = pow(10, -3)
    m._epsilon_2 = pow(10, -2)
    m._cuts = set()
    m._n = n

    obj = quicksum(d[i, j] * x[i, j] for (i, j) in x.keys())
    m.setObjective(obj, GRB.MINIMIZE)

    if DL:
        # D-L constraints
        u = {}
        for i in V - {0}:
            u[i] = m.addVar()
            m.addLConstr(u[i] >= 3 + (n - 3) * x[i, 0] - x[0, i])
            m.addLConstr(u[i] <= n - 1 - (n - 3) * x[0, i] + x[i, 0])
        for (i, j) in Arcs_no_depot:
            m.addLConstr(u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2)
    if GG:
        # G-G constraints
        f = {}
        for (i, j) in Arcs:
            f[i, j] = m.addVar()
        if GG_modified:
            for (i, j) in Arcs_no_depot:
                m.addLConstr(f[i, j] <= (n - 3) * x[i, j])
            for i in V - {0}:
                m.addLConstr((n - 2) * x[0, i] + quicksum(f[j, i] for j in V - {0, i}) == 1 - x[i, 0] + quicksum(f[i, j] for j in V - {0, i}))
        else:
            for (i, j) in Arcs:
                m.addLConstr(f[i, j] <= (n - 1) * x[i, j])
            for i in V - {0}:
                m.addLConstr(quicksum(f[j, i] for j in V - {i}) == 1 + quicksum(f[i, j] for j in V - {0, i}))
    if GG and DL:
        # G-G+D-L constraints
        for i in V - {0}:
            if GG_modified:
                m.addLConstr(u[i] + quicksum(f[i, j] for j in V - {0, i}) - x[i, 0] == n - 1)
            else:
                m.addLConstr(u[i] == n - quicksum(f[i, j] for j in V - {0, i}))

    start_time = time()
    m.optimize()

    if add_root_node_cuts:
        new_bound, iteration_counter, execution_time = separate_DFJ_cuts(m, n)
        print('new ATSP bound: ', new_bound)
        print('DFJ cuts identified, ', len(m._cuts), ' CPU:', round(execution_time, 1))
        remove_nonbinding_DFJ_cuts(m, m._cuts)

    model_string = ''
    if GG:
        model_string = 'GG'
    if GG_modified:
        model_string = model_string + ' m.'
    if DL:
        model_string = model_string + ' DL'
    print('Instance: ', filename)
    print(model_string + ' Original ATSP bound: ', round(m.objval, 2))

    if not relax:
        for (i, j) in x.keys():
            x[i, j].vtype = GRB.BINARY
        m.setParam('MIPGap', 0.0)
        m.setParam('MIPGapAbs', 0.99)
        m.setParam('OutputFlag', True)
        m.setParam('TimeLimit', timelimit)
        m.setParam('DisplayInterval', 100)

        def DFJ_callback(model, where):
            if where == GRB.Callback.MIPNODE:  # fractional node
                # add cuts at every bnb node
                if model.cbGet(GRB.callback.MIPNODE_NODCNT) % 3 == 0:  # frequency control
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:  # only search for cuts in feasible nodes
                        x_values = model.cbGetNodeRel(model._x)
                        violated_DFJ_cuts = find_DFJ_cuts(model._n, x_values, model._epsilon_1, model._epsilon_2)
                        for S in violated_DFJ_cuts:
                            S_code = set_to_code(S, n)
                            #print(S_code, S)
                            if S_code not in m._cuts:
                                m._cuts.add(S_code)
                                Constr_LHS = quicksum(model._x[i, j] for (i, j) in product(S, S) if i != j)
                                model.cbCut(Constr_LHS, GRB.LESS_EQUAL, len(S) - 1)
        m.optimize(DFJ_callback)
        #m.optimize()
        print('Gap: ', round(100 * m.MIPGap, 2))
    print('status: %s route length %g' % (m.status, m.objval))
    print('CPU: ', round(time() - start_time, 1))
    print()
#========================================================================================================
def main_ATSP():
    relax = True
    GG = True
    GG_modified = True
    DL = True
    add_root_node_cuts = False
    timelimit = 3600
    GG_DL_ATSP('realfiles', 'dc112', 112, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc126', 126, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc134', 134, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc176', 176, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc188', 188, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc563', 563, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc849', 849, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc895', 895, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
    GG_DL_ATSP('realfiles', 'dc932', 932, relax, GG, GG_modified, DL, add_root_node_cuts, timelimit)
#========================================================================================================
if __name__ == "__main__":
    main_ATSP()