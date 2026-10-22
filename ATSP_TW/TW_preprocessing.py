from itertools import product
#-----------------------------------------------------------------------------------------------------------------------
def update_TW_parameters(C, depot, Tau, a, b, Eliminated_Arcs, T, tau_tilde_from_depot, tau_tilde_to_depot):
    infinity = pow(10, 9)
    Arcs_no_depot = [(i, j) for (i, j) in product(C - {depot}, C - {depot}) if i != j]
    Preprocessing_continue = True
    while Preprocessing_continue:
        updated_tw_bounds = 0
        for i in C - {depot}:
            arcs_to_i = {j for j in C - {depot, i} if [j, i] not in Eliminated_Arcs}
            if len(arcs_to_i) > 0:
                new_a_i_no_depot = min(a[j] + Tau[j, i] for j in arcs_to_i)
                new_a_i = min(new_a_i_no_depot, tau_tilde_from_depot[i])
                if new_a_i >= a[i] + 0.0001:
                    # print('earlier time 1 ', i, a[i], new_a_i)
                    a[i] = new_a_i
                    updated_tw_bounds += 1
        for i in C - {depot}:
            arcs_from_i = {j for j in C - {depot, i} if [i, j] not in Eliminated_Arcs}
            if len(arcs_from_i) > 0:
                if T < infinity:
                    new_a_i = min(min(a[j] - Tau[i, j] for j in arcs_from_i), T - tau_tilde_to_depot[i])
                else:
                    new_a_i = min(a[j] - Tau[i, j] for j in arcs_from_i)
                if new_a_i >= a[i] + 0.0001:
                    # print('earlier time 2 ', i, a[i], new_a_i)
                    a[i] = new_a_i
                    updated_tw_bounds += 1
        for i in C - {depot}:
            arcs_to_i = {j for j in C - {depot, i} if [j, i] not in Eliminated_Arcs}
            if len(arcs_to_i) > 0:
                new_b_i = max(a[i], max(b[j] + Tau[j, i] for j in arcs_to_i))
                if new_b_i <= b[i] - 0.0001:
                    # print('later time 2 ', i, new_b_i, b[i])
                    b[i] = new_b_i
                    updated_tw_bounds += 1
        for i in C - {depot}:
            arcs_from_i = {j for j in C - {depot, i} if [i, j] not in Eliminated_Arcs}
            if len(arcs_from_i) > 0:
                new_b_i_no_depot = max(b[j] - Tau[i, j] for j in arcs_from_i)
                if T < infinity:
                    new_b_i_depot = T - tau_tilde_to_depot[i]
                    new_b_i = max(new_b_i_no_depot, new_b_i_depot)
                else:
                    new_b_i = new_b_i_no_depot
                if new_b_i <= b[i] - 0.0001:
                    # print('later time 1 ', i, new_b_i, b[i])
                    b[i] = new_b_i
                    updated_tw_bounds += 1
        new_eliminated_arcs = 0
        for (i, j) in Arcs_no_depot:
            if a[j] + Tau[j, i] >= b[i] + 0.0001 and [j, i] not in Eliminated_Arcs:
                Eliminated_Arcs.append([j, i])
                new_eliminated_arcs += 1
            if min(b[i], b[j] - Tau[i, j]) <= a[i] and [i, j] not in Eliminated_Arcs:
                Eliminated_Arcs.append([i, j])
                new_eliminated_arcs += 1
        if updated_tw_bounds == 0 and new_eliminated_arcs == 0:
            Preprocessing_continue = False
        else:
            print('# tightened time bounds: ', updated_tw_bounds, '# eliminated arcs: ', new_eliminated_arcs)
    return a, b, Eliminated_Arcs