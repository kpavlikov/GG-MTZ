import numpy as np
#import tsplib95
import os
#-----------------------------------------------------------------------------------------------------------------------
def load_instance(path, filename, n, symmetry_flag, data_format_flag):
    #path = os.path.dirname(__file__)
    #path = os.path.dirname(os.path.dirname(__file__))

    d = np.zeros((n, n))
    if data_format_flag == 'CONCORD':
        if path != '':
            temp = np.loadtxt(path + '/' + filename + ".txt")
        else:
            temp = np.loadtxt(filename + ".txt")
        if symmetry_flag:
            for i in range(0, n):
                for j in range(0, n):
                    if i < j:
                        d[i, j] = np.round(np.sqrt(np.square(temp[i, 1] - temp[j, 1]) + np.square(temp[i, 2] - temp[j, 2])))
                        d[j, i] = d[i, j]
    if data_format_flag == 'TSPLIB':
        if symmetry_flag == 'yes':
            problem = tsplib95.load_problem(path + '/' + filename + ".tsp")
        else:
            problem = tsplib95.load_problem(path + '/' + filename + ".atsp")
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    d[i, j] = problem.wfunc(i, j)
    if data_format_flag == 'SQUARE':
        if not symmetry_flag:
            temp = np.loadtxt(path + '/' + filename + ".txt")
            for i in range(0, n):
                for j in range(0, n):
                    if i != j:
                        d[i, j] = temp[i, j]
    if data_format_flag == 'VECTOR':
        temp = np.loadtxt(path + '/' + filename + ".txt")
        for i in range(0, len(temp)):
            j = i % n
            d[int((i - j)/n), j] = temp[i]
    symmetry_check = 1
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                if d[i, j] != d[j, i]:
                    symmetry_check = 0
                    break
    print()
    print('#####################################################')
    if symmetry_check == 1:
        print('Symmetric instance loaded ' + filename)
    else:
        print('Asymmetric instance loaded ' + filename)
        asymmtery_measure = 0
        epsilon = 0
        D_sum = 0
        for i in range(0, n):
            for j in range(0, n):
                if i < j:
                    D_sum += d[i, j] + d[j, i]
                    if d[i, j] != d[j, i]:
                        asymmtery_measure += 1
                        epsilon += abs(d[i, j] - d[j, i])
        print('Asymmetry measure Type 1:', round(100 * 2 * asymmtery_measure / (n * (n - 1)), 2), '%')
        print('Asymmetry measure Type 2:', round(100 * epsilon / D_sum, 2), '%')
        #for i in range(0, n):
        #    count = 0
        #    for j in range(0, n):
        #        if i != j and d[i, j] != d[j, i]:
        #            count += 1
        #    if count > 0:
        #        print('# of asymmetric arcs from node ', i, ' is ', count)
    return d
#-----------------------------------------------------------------------------------------------------------------------
def load_instance_symmetrize(path, filename, n, symmetry_flag, data_format_flag):
    #path = os.path.dirname(__file__)
    #path = os.path.dirname(os.path.dirname(__file__))

    # GEO distances
    #temp = np.loadtxt(filename)
    #lat = np.zeros(n)
    #long = np.zeros(n)
    #PI = 3.14159265358
    #R = 6378.388
    #for i in V:
    #    deg = round(temp[i, 1])
    #    min = temp[i, 1] - deg;
    #    lat[i] = PI * (deg + 5.0 * min / 3.0) / 180.0;
    #    deg = round(temp[i, 2])
    #    min = temp[i, 2] - deg;
    #    long[i] = PI * (deg + 5.0 * min / 3.0) / 180.0;
    #for (i, j) in Arcs:
    #    q1 = cos(long[i] - long[j])
    #    q2 = cos(lat[i] - lat[j])
    #    q3 = cos(lat[i] + lat[j])
    #    d[i, j] = np.round(R * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)))
    #    d[j, i] = d[i, j]

    d = np.zeros((n, n))
    if data_format_flag == 'TSPLIB':
        problem = tsplib95.load_problem(path + '/' + filename + ".atsp")
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    d[i, j] = problem.wfunc(i, j)
    if data_format_flag == 'SQUARE':
        if symmetry_flag == 'no':
            temp = np.loadtxt(path + '/' + filename + ".txt")
            for i in range(0, n):
                for j in range(0, n):
                    if i != j:
                        d[i, j] = temp[i, j]
    if data_format_flag == 'VECTOR':
        temp = np.loadtxt(path + '/' + filename + ".txt")
        for i in range(0, len(temp)):
            j = i % n
            d[int((i - j)/n), j] = temp[i]
    symmetry_check = 1
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                if d[i, j] != d[j, i]:
                    symmetry_check = 0
                    break
    if symmetry_check == 1:
        print('Symmetric instance loaded ' + filename)
    else:
        print('Asymmetric instance loaded ' + filename)
        asymmtery_measure = 0
        epsilon = 0
        D_sum = 0
        for i in range(0, n):
            for j in range(0, n):
                if i < j:
                    D_sum += d[i, j] + d[j, i]
                    if d[i, j] != d[j, i]:
                        asymmtery_measure += 1
                        epsilon += abs(d[i, j] - d[j, i])
        print('Asymmetry measure Type 1:', round(100 * 2 * asymmtery_measure / (n * (n - 1)), 2), '%')
        print('Asymmetry measure Type 2:', round(100 * epsilon / D_sum, 2), '%')
        for i in range(0, n):
            count = 0
            for j in range(0, n):
                if i != j and d[i, j] != d[j, i]:
                    count += 1
            print('# of asymmetric arcs from node ', i, ' is ', count)
        D_tilde, N = symmetrize_distance_matrix(n, d)

    return D_tilde, N
#-----------------------------------------------------------------------------------------------------------------------
def Find_triangle(n, d):
    sum = 0
    for k in range(0, n):
        for i in range(0, n):
            for j in range(0, n):
                if k != i and i != j and j != k:
                    if d[k, i] + d[i, j] + d[j, k] > sum:
                        sum = d[k, i] + d[i, j] + d[j, k]
                        index_1 = k
                        index_2 = i
                        index_3 = j
    return index_1, index_2, index_3
#-----------------------------------------------------------------------------------------------------------------------
def symmetrize_distance_matrix(n, D):
    print(D)
    S = set()
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                if D[i, j] != D[j, i]:
                    S.add(i)
                    S.add(j)
    print(print('Size of asymmetry set: ', len(S)))
    N = len(S) + n
    print(list(S))
    #for i in range(0, len(S)):
    #    print(list(S)[i])
    N = len(S) + n
    D_tilde = pow(10, 10) * np.ones((N, N))
    for i in range(0, len(S)):
        for j in range(0, len(S)):
            if i != j:
                D_tilde[i, j + len(S)] = D[list(S)[j], list(S)[i]]
                D_tilde[i + len(S), j] = D[list(S)[i], list(S)[j]]
            else:
                D_tilde[i, j + len(S)] = 2 * pow(10, 10)# 0
                D_tilde[j + len(S), i] = 2 * pow(10, 10)# 0
    print(n - len(S))
    index = 0
    for i in range(0, n):
        if i not in S:
            for j in range(0, len(S)):
                D_tilde[j, index + 2 * len(S)] = D[i, list(S)[j]]
                D_tilde[index + 2 * len(S), j] = D[i, list(S)[j]]
                D_tilde[j + len(S), index + 2 * len(S)] = D[i, list(S)[j]]
                D_tilde[index + 2 * len(S), j + len(S)] = D[i, list(S)[j]]
            index += 1
    index1 = 0
    for i in range(0, n):
        index2 = 0
        if i not in S:
            for j in range(0, n):
                if j not in S:
                    D_tilde[index1 + 2 * len(S), index2 + 2 * len(S)] = D[i, j]
                    index2 += 1
            index1 += 1
    #print(D_tilde)
    return D_tilde, N
#-----------------------------------------------------------------------------------------------------------------------
def convert_distance_TSPLIB(path, filename, n, symmetry_flag, data_format_flag):
    d = load_instance(path, filename, n, symmetry_flag, data_format_flag)
    f = open(path + '/' + filename + ".atsp", "w+")
    for i in range(0, n):
        s = ''
        for j in range(0, n):
            s = s + str(d[i, j]) + ' '
        f.write(s + '\n')
    f.close()
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def add_starting_solution(model, x, n, d, folder, filename, LKH_start, symmetry_flag):
    if LKH_start:
        solution = np.loadtxt(folder + '/' + filename + '_solution.txt')
        if len(solution) > 1:
            UB = solution[0]
            for k in range(1, n):
                model._x[round(solution[k]), round(solution[k + 1])].start = 1
    print('UB: ', UB)
    return(UB)
#convert_distance_TSPLIB('realfiles', 'dc112', 112, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc126', 126, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc134', 134, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc176', 176, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc188', 188, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc563', 563, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc849', 849, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc895', 895, 'no', 'VECTOR')
#convert_distance_TSPLIB('realfiles', 'dc932', 932, 'no', 'VECTOR')
#-----------------------------------------------------------------------------------------------------------------------
def dec_to_code(number):
    """Converts a non-negative number to a list of digits in the given base.

    The base must be an integer greater than or equal to 2 and the first digit
    in the list of digits is the most significant one.
    """
    if not number:
        return [0]
    base = 36
    digits = []
    while number:
        digits.append(number % base)
        number //= base
    digits_letters = []
    number_string = ''
    for i in reversed(digits):
        if i <= 9:
            digits_letters.append(i)
            number_string += str(i)
        elif i == 10:
            digits_letters.append('a')
            number_string += 'a'
        elif i == 11:
            digits_letters.append('b')
            number_string += 'b'
        elif i == 12:
            digits_letters.append('c')
            number_string += 'c'
        elif i == 13:
            digits_letters.append('d')
            number_string += 'd'
        elif i == 14:
            digits_letters.append('e')
            number_string += 'e'
        elif i == 15:
            digits_letters.append('f')
            number_string += 'f'
        elif i == 16:
            digits_letters.append('g')
            number_string += 'g'
        elif i == 17:
            digits_letters.append('h')
            number_string += 'h'
        elif i == 18:
            digits_letters.append('i')
            number_string += 'i'
        elif i == 19:
            digits_letters.append('j')
            number_string += 'j'
        elif i == 20:
            digits_letters.append('k')
            number_string += 'k'
        elif i == 21:
            digits_letters.append('l')
            number_string += 'l'
        elif i == 22:
            digits_letters.append('m')
            number_string += 'm'
        elif i == 23:
            digits_letters.append('n')
            number_string += 'n'
        elif i == 24:
            digits_letters.append('o')
            number_string += 'o'
        elif i == 25:
            digits_letters.append('p')
            number_string += 'p'
        elif i == 26:
            digits_letters.append('q')
            number_string += 'q'
        elif i == 27:
            digits_letters.append('r')
            number_string += 'r'
        elif i == 28:
            digits_letters.append('s')
            number_string += 's'
        elif i == 29:
            digits_letters.append('t')
            number_string += 't'
        elif i == 30:
            digits_letters.append('u')
            number_string += 'u'
        elif i == 31:
            digits_letters.append('v')
            number_string += 'v'
        elif i == 32:
            digits_letters.append('w')
            number_string += 'w'
        elif i == 33:
            digits_letters.append('x')
            number_string += 'x'
        elif i == 34:
            digits_letters.append('y')
            number_string += 'y'
        elif i == 35:
            digits_letters.append('z')
            number_string += 'z'
        #print(i)
    #return list(reversed(digits_letters))
    return number_string
#-----------------------------------------------------------------------------------------------------------------------
def code_to_dec(code):
    """Converts a list of digits in the given base to an integer.

    The first digit is the most significant and the base is assumed to
    be an integer greater than or equal to 2.
    """
    base = 36
    power = 1
    ttt = 0
    #for digit in reversed(digits):
    for t in reversed(range(len(code))):
        i = code[t]
        if i in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
            number = int(i)
        elif i == 'a':
            number = 10
        elif i == 'b':
            number = 11
        elif i == 'c':
            number = 12
        elif i == 'd':
            number = 13
        elif i == 'e':
            number = 14
        elif i == 'f':
            number = 15
        elif i == 'g':
            number = 16
        elif i == 'h':
            number = 17
        elif i == 'i':
            number = 18
        elif i == 'j':
            number = 19
        elif i == 'k':
            number = 20
        elif i == 'l':
            number = 21
        elif i == 'm':
            number = 22
        elif i == 'n':
            number = 23
        elif i == 'o':
            number = 24
        elif i == 'p':
            number = 25
        elif i == 'q':
            number = 26
        elif i == 'r':
            number = 27
        elif i == 's':
            number = 28
        elif i == 't':
            number = 29
        elif i == 'u':
            number = 30
        elif i == 'v':
            number = 31
        elif i == 'w':
            number = 32
        elif i == 'x':
            number = 33
        elif i == 'y':
            number = 34
        elif i == 'z':
            number += 35
        else:
            print(number)
        ttt += power * number
        power *= base
    return ttt
#-----------------------------------------------------------------------------------------------------------------------
def S_to_bin(S, n):
    code = ''
    for i in range(0, n):
        if n - i in S:
            code += '1'
        else:
            code += '0'
    for i in range(0, n):
        if code[i] == '1':
            code = code[i:]
            break
    return code
#-----------------------------------------------------------------------------------------------------------------------
def bin_to_dec(code):
    code = '0b'+code
    return(int(code, 2))
#-----------------------------------------------------------------------------------------------------------------------
def bin_to_set(code):
    code = code[2:]
    S = set()
    n = len(code)
    for t in reversed(range(len(code))):
        if code[t] == '1':
            S.add(n - t)
    return S
#-----------------------------------------------------------------------------------------------------------------------
def set_to_code(S, n):
    S_bin = S_to_bin(S, n)
    S_dec = bin_to_dec(S_bin)
    code = dec_to_code(S_dec)
    return code
#-----------------------------------------------------------------------------------------------------------------------
def code_to_set(code):
    code_in_dec = code_to_dec(code)
    code_in_bin = bin(code_in_dec)
    S = bin_to_set(code_in_bin)
    return S
#-----------------------------------------------------------------------------------------------------------------------
# S = {3, 4, 12, 20}
# n = 25
# print(S, set_to_code(S, n))
# print(set_to_code(S, n), code_to_set(set_to_code(S, n)))
