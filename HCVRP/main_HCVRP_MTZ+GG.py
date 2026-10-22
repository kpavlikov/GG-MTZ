from RCI_lib import *
from BBV_RCI_lib import *
from FCI_lib_new import *
from time import *
from utilities import *
from mip import ConstrsGenerator, CutPool
#import sys
#sys.path.insert(1, 'C:\Users\sa-kop\Documents\HCVRP')
import sys
# #=======================================================================================================================
# class Logger(object):
#     def __init__(self):
#         self.terminal = sys.stdout
#         self.log = open("log.dat", "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#=======================================================================================================================
def HCVRP_Yaman_formulation(folder, filename, CVRP_UB, n, Q, F, mu, relax, timelim, GG, m, DL, lazy, need_to_run_RCI_separation, need_to_run_BBV_RCI_separation, need_to_run_FCI_separation):
    #sys.stdout = Logger()
    C = set(range(1, n + 1))
    C_0 = set(range(0, n + 1))
    Edges = [(i, j) for (i, j) in product(C_0, C_0) if i < j]
    #Edges_no_origin = [(i, j) for (i, j) in product(C, C) if i < j]
    Arcs = {(i, j) for (i, j) in product(C_0, C_0) if i != j}
    Arcs_no_origin = {(i, j) for (i, j) in product(C, C) if i != j}
    VRP_data = np.loadtxt(folder + '/' + filename + '.txt')
    d = {}
    demand = VRP_data[:, 3]
    for (i, j) in Edges:
        d[i, j] = np.round(np.sqrt(np.square(VRP_data[i, 1] - VRP_data[j, 1]) + np.square(VRP_data[i, 2] - VRP_data[j, 2])))
    total_demand = sum(demand[i] for i in C)
    if mu is None:
        mu = np.zeros(len(Q))
    for k in range(0, len(Q)):
        tmp = bin_pack_S(k, C, Q, demand)
        if tmp < mu[k] or mu[k] == 0:
            mu[k] = tmp
    mu = [round(mu[k]) for k in range(len(Q))]
    min_capacity_for_C = min_capacity_cover_for_S(C, Q, None, mu, demand)
    print()
    print('total demand: ', total_demand, min_capacity_for_C)
    print('Q:', Q, ' mu:', mu)
    K = len(Q)

    #HCVRP_model = Model(solver_name=CBC)
    HCVRP_model = Model(solver_name=GRB)
    HCVRP_model._n = n
    HCVRP_model._K = K
    HCVRP_model._C = C
    HCVRP_model._Q = Q
    HCVRP_model._demand = demand
    HCVRP_model._mu = mu
    HCVRP_model.verbose = 0
    epsilon_1 = 0.0001
    epsilon_2 = 0.005
    epsilon_3 = 0.01
    HCVRP_model._epsilon_1 = epsilon_1
    HCVRP_model._epsilon_2 = epsilon_2
    HCVRP_model._epsilon_3 = epsilon_3
    x = {}
    a = {}
    for i in C:
        for k in range(0, K):
            a[i, k] = HCVRP_model.add_var()
            if Q[k] < demand[i] - 0.5:
                HCVRP_model += a[i, k] == 0
    for k in range(0, K):
        HCVRP_model += xsum(a[i, k] for i in C) <= mu[k]
    # binary variables x[i, j] indicating if arc {i, j} is used or not
    for (i, j) in Arcs:
        x[i, j] = HCVRP_model.add_var()

    HCVRP_model._RCI_counter = 0
    HCVRP_model._BBV_counter = 0
    HCVRP_model._FCI_counter = 0

    for i in C:
        HCVRP_model += x[i, 0] == xsum(a[i, k] for k in range(0, K))
        # each city degree constraint: the city is visited only once
        HCVRP_model += xsum(x[i, j] for j in C_0 - {i}) == 1
        HCVRP_model += xsum(x[j, i] for j in C_0 - {i}) == 1
    if DL:
        u = {}
        for i in C:
            u[i] = HCVRP_model.add_var()
            HCVRP_model += u[i] >= demand[i] + xsum(demand[j] * x[j, i] for j in C - {i})
            # (40) by Yaman
            K_i = {k for k in range(0, K) if Q[k] >= demand[i]}
            y_0_i_coefficient = min(Q[0] - demand[i] - max(demand[j] for j in C - {i}), min(Q[k] for k in K_i) - demand[i])
            HCVRP_model += u[i] <= xsum(Q[k] * a[i, k] for k in K_i) + Q[0] * (1 - xsum(a[i, k] for k in K_i)) \
                                 - xsum(demand[j] * x[i, j] for j in C - {i}) - x[0, i] * y_0_i_coefficient
        if not lazy:
            Yaman_constraints = {}
            for (i, j) in Arcs_no_origin:
                # DL adaptation to HCVRP by Yaman, equation (38)
                # (39)
                K_i = {k for k in range(0, K) if Q[k] >= demand[i]}
                K_i_j = {k for k in K_i if Q[k] >= demand[i] + demand[j]}
                # (40)
                #Yaman_constraints[i, j] = u[j] - u[i] - demand[j] + Q[0] * (1 - x[i, j]) - xsum((Q[0] - Q[k]) * a[i, k] for k in K_i) - x[j, i] * (min(Q[k] for k in K_i_j) - demand[i] - demand[j])
                Yaman_constraints[i, j] = u[j] - u[i] - demand[j] + Q[0] * (1 - x[i, j]) - xsum((Q[0] - Q[k]) * a[i, k] for k in K_i) - x[j, i] * max(0, (min(Q[k] for k in K_i_j) - demand[i] - demand[j]))
                #HCVRP_model += u[j] >= u[i] + demand[j] - Q[0] * (1 - x[i, j]) + xsum((Q[0] - Q[k]) * a[i, k] for k in K_i) + x[j, i] * (min(Q[k] for k in K_i_j) - demand[i] - demand[j])
                if not (GG and lazy):
                    HCVRP_model += Yaman_constraints[i, j] >= 0
    if GG:
        f = {}
        for (i, j) in Arcs_no_origin:
            f[i, j] = HCVRP_model.add_var()
        for i in C:
            f[i, 0] = HCVRP_model.add_var()
        inflow = {}
        outflow = {}
        if m:
            for (i, j) in Arcs_no_origin:
                HCVRP_model += f[i, j] <= (Q[0] - demand[i] - demand[j]) * x[i, j]
            for i in C:
                HCVRP_model += f[i, 0] + demand[i] * x[i, 0] <= xsum(Q[k] * a[i, k] for k in range(0, K))
                inflow[i] = xsum(f[j, i] + demand[j] * x[j, i] for j in C - {i})
                outflow[i] = xsum(f[i, j] + demand[i] * x[i, j] for j in C_0 - {i})
        else:
            for (i, j) in Arcs_no_origin:
                HCVRP_model += f[i, j] <= (Q[0] - demand[j]) * x[i, j]
            for i in C:
                HCVRP_model += f[i, 0] <= xsum(Q[k] * a[i, k] for k in range(0, K))
                inflow[i] = xsum(f[j, i] for j in C - {i})
                outflow[i] = xsum(f[i, j] for j in C_0 - {i})
        for i in C:
            HCVRP_model += inflow[i] + demand[i] == outflow[i]
    if GG and DL:
        for i in C:
            HCVRP_model += u[i] == outflow[i]

    # objective function: minimize the total cost
    HCVRP_model.objective = minimize(xsum(F[k] * a[i, k] for i in C for k in range(0, K)) + xsum(d[i, j] * (x[i, j] + x[j, i]) for (i, j) in Edges))

    HCVRP_model.verbose = 0
    HCVRP_model.optimize()
    print('Initial LP relaxation:', round(HCVRP_model.objective_value, 2))
    add_capacity_constraints_for_C = True
    if add_capacity_constraints_for_C:
        # RC inequality for S = C
        RCI_LHS, RCI_RHS = create_RCI(x, C, C, Q, mu, demand)
        HCVRP_model += RCI_LHS <= RCI_RHS
        # BBV-RC inequalities for S = C
        for k in range(0, K - 1):
             cut_LHS, cut_RHS = create_BBV_RCI(k, x, a, C, C, Q, demand)
             HCVRP_model += cut_LHS <= cut_RHS
        # strong FC inequalities for S = C
        create_strong_FCI(HCVRP_model, x, a, C)
        HCVRP_model.optimize()
        print('LP relaxation after adding default RC inequalities:', round(HCVRP_model.objective_value, 2))
    print()

    run_separation = need_to_run_RCI_separation or need_to_run_BBV_RCI_separation or need_to_run_FCI_separation
    if run_separation:
        separation_start_time = time()
        while run_separation:
            current_total_number_of_cuts = HCVRP_model._RCI_counter + HCVRP_model._BBV_counter + HCVRP_model._FCI_counter
            if need_to_run_RCI_separation:
                RCI_Sep(HCVRP_model, x)
                print(filename + ' K=' + str(K) + ' Q=' + str(Q) + ' mu=' + str(mu) + ' HCVRP status: %s route length %g' % (HCVRP_model.status, HCVRP_model.objective_value))
            if need_to_run_BBV_RCI_separation:
                BBV_RCI_Sep(HCVRP_model, x, a)
                print(filename + ' K=' + str(K) + ' Q=' + str(Q) + ' mu=' + str(mu) + ' HCVRP status: %s route length %g' % (HCVRP_model.status, HCVRP_model.objective_value))
            if need_to_run_FCI_separation:
                FCI_Sep(HCVRP_model, x, a)
                print(filename + ' K=' + str(K) + ' Q=' + str(Q) + ' mu=' + str(mu) + ' HCVRP status: %s route length %g' % (HCVRP_model.status, HCVRP_model.objective_value))
            if HCVRP_model._RCI_counter + HCVRP_model._BBV_counter + HCVRP_model._FCI_counter == current_total_number_of_cuts:
                run_separation = False
        remove_nonbinding_RCI(HCVRP_model)
        remove_nonbinding_BBV(HCVRP_model)
        remove_nonbinding_FCI(HCVRP_model)
        print('Separation CPU time: ', round(time() - separation_start_time, 1))
    if relax:
        start_time = time()
        status = HCVRP_model.optimize()
        print('Objective: ', np.round(HCVRP_model.objective_value, 2))
        print('CPU: ', round(time() - start_time, 1))
        print(filename + ' K=' + str(K) + ' Q=' + str(Q) + ' mu=' + str(mu) + ' HCVRP status: %s' % (HCVRP_model.status))
    else:
        make_binary(a)
        make_binary(x)
        for i in C:
            x[i, 0].__setattr__('var_type', CONTINUOUS)
        HCVRP_model.max_mip_gap = 0.00
        Distance_matrix_integral = True
        for (i, j) in d.keys():
            if abs(d[i, j] - round(d[i, j])) >= 0.0001:
                Distance_matrix_integral = False
                break
        if Distance_matrix_integral:
            HCVRP_model.max_mip_gap_abs = 0.99

        mip_start_time = time()
        if GG and DL and lazy:
            class Yaman_CutGenerator(ConstrsGenerator):
                """Class to add Yaman constraints in the lazy fashion"""
                def __init__(self, x, u, a):
                    self.x = x
                    self.u = u
                    self.a = a
                def generate_constrs(self, HCVRP_model: Model, depth: int = 0, npass: int = 0):
                    x, u, a, cp = HCVRP_model.translate(self.x), HCVRP_model.translate(self.u), HCVRP_model.translate(self.a), CutPool()
                    for (i, j) in Arcs_no_origin - DL_cuts_capacity:
                        K_i = {k for k in range(0, K) if Q[k] >= demand[i]}
                        K_i_j = {k for k in K_i if Q[k] >= demand[i] + demand[j]}
                        Yaman_constraint = u[j] - u[i] - demand[j] + Q[0] * (1 - x[i, j]) - \
                                           xsum((Q[0] - Q[k]) * a[i, k] for k in K_i) - \
                                           x[j, i] * max(0, (min(Q[k] for k in K_i_j) - demand[i] - demand[j]))
                        if Yaman_constraint.x <= -0.0001:
                            DL_cuts_capacity.add((i, j))
                            cp.add(Yaman_constraint >= 0)
                    for cut in cp.cuts:
                        HCVRP_model += cut
            DL_cuts_capacity = set()
            HCVRP_model.cuts_generator = Yaman_CutGenerator(x, u, a)
        #HCVRP_model.verbose = 1
        status = HCVRP_model.optimize(timelim)
        #status = model.optimize(max_nodes=1)
        if HCVRP_model.objective_value and status != OptimizationStatus.INFEASIBLE:
            print('Objective: ', HCVRP_model.objective_value)
            print('Bound: ', HCVRP_model.objective_bound)
            print('Gap: ', round(HCVRP_model.gap * 100, 2), '%')
            print('CPU: ', round(time() - mip_start_time, 1))
            if lazy:
                print('Lazy cuts imposed: ', len(DL_cuts_capacity))
            print(filename + ' K=' + str(K) + ' Q=' + str(Q) + ' mu=' + str(mu) + ' HCVRP status: %s' % (HCVRP_model.status))
        else:
            print('The problem is infeasible.')
#=======================================================================================================================
def main_HCVRP(relax, GG, m, DL, lazy, timelim):
    separate_RCI = True
    separate_RCI_BBV = True
    separate_FCI = True
    # problem 3
    n = 20
    Capacities = [120, 70, 40, 30, 20]
    Fixed_costs = [225, 120, 50, 35, 20]
    available_vehicles = 2 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 4
    n = 20
    Capacities = [150, 80, 60]
    Fixed_costs = [3000, 1500, 1000]
    available_vehicles = 2 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 5
    n = 20
    Capacities = [120, 70, 40, 30, 20]
    Fixed_costs = [225, 120, 50, 35, 20]
    available_vehicles = 2 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8_new_depot', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8_new_depot', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 6
    n = 20
    Capacities = [150, 80, 60]
    Fixed_costs = [3000, 1500, 1000]
    available_vehicles = 3 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8_new_depot', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8_new_depot', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 13
    n = 50
    Capacities = [200, 120, 70, 40, 30, 20]
    Fixed_costs = [400, 225, 120, 50, 35, 20]
    available_vehicles = 3 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 14
    n = 50
    Capacities = [300, 160, 120]
    Fixed_costs = [3500, 1500, 1000]
    available_vehicles = 3 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 15
    n = 50
    Capacities = [160, 100, 50]
    Fixed_costs = [450, 250, 100]
    available_vehicles = 3 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 16
    n = 50
    Capacities = [140, 80, 40]
    Fixed_costs = [400, 200, 100]
    available_vehicles = 4 * np.ones(len(Capacities))
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    HCVRP_Yaman_formulation('CE_1969', 'instance_8', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 17
    n = 75
    Capacities = [350, 200, 120, 50]
    Fixed_costs = [320, 150, 80, 25]
    available_vehicles = 4 * np.ones(len(Capacities))
    #HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    #HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 18
    n = 75
    Capacities = [400, 250, 150, 100, 50, 20]
    Fixed_costs = [800, 400, 180, 100, 35, 10]
    available_vehicles = 3 * np.ones(len(Capacities))
    #HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    #HCVRP_Yaman_formulation('CE_1969', 'instance_9', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 19
    n = 100
    Capacities = [300, 200, 100]
    Fixed_costs = [2100, 1200, 500]
    available_vehicles = 4 * np.ones(len(Capacities))
    #HCVRP_Yaman_formulation('CE_1969', 'instance_10', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    #HCVRP_Yaman_formulation('CE_1969', 'instance_10', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)

    # problem 20
    n = 100
    Capacities = [200, 140, 60]
    Fixed_costs = [500, 300, 100]
    available_vehicles = 4 * np.ones(len(Capacities))
    #HCVRP_Yaman_formulation('CE_1969', 'instance_10', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
    available_vehicles = None
    #HCVRP_Yaman_formulation('CE_1969', 'instance_10', None, n, Capacities, Fixed_costs, available_vehicles, relax, timelim, GG, m, DL, lazy, separate_RCI, separate_RCI_BBV, separate_FCI)
#=======================================================================================================================
def main():
    timelim = 3600
    relax = False
    lazy = False
    GG = True
    m = False
    DL = False
    #main_HCVRP(relax, GG, m, DL, lazy, timelim)
    GG = True
    m = True
    DL = False
    #main_HCVRP(relax, GG, m, DL, lazy, timelim)
    GG = False
    m = False
    DL = True
    #main_HCVRP(relax, GG, m, DL, lazy, timelim)
    GG = True
    m = True
    DL = True
    main_HCVRP(relax, GG, m, DL, lazy, timelim)

    lazy = True
    #main_HCVRP(relax, GG, m, DL, lazy, timelim)
#=======================================================================================================================
if __name__ == "__main__":
    main()