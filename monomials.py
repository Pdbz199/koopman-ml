# import sympy.abc
# from sympy.utilities.lambdify import lambdify, implemented_function

# x = 1
# y = 2
# x_str = ""
# for i in range(x.shape[0]):
#     x_str += '{' + str(i) + '}, '
# x_syms = symbols(x_str)
# symbol_to_value = {}
# for i, sym in enumerate(x_syms):
#     symbol_to_value[sym] = x[i]
# print(symbol_to_value)
# M = itermonomials(x_syms, 2)
# sortedM = sorted(M, key=monomial_key('grlex', np.flip(x_syms)))
# print(sortedM)

# for i in range(len(sortedM)):
#     substitutes = [0 for i in range(x.shape[0])]
#     print("expression", sortedM[i])
#     for sym in sorted(sortedM[i].free_symbols, key=lambda symbol: symbol.name):
#         substitutes[ind] = 'np.array(' + ','.join('{}'.format(symbol_to_value[sym]).split(' ')) + ')'

#     print('substitutes', substitutes)
#     substituted = str(sortedM[i]).format(*substitutes)
#     print("substituted", substituted)
#     print("evaluated", eval(substituted))

# print(sympy.diff(f, sympy.abc.x))

# applies sympy function with given value
# def f_subs(x):
#     print(x)
#     return f.subs(sympy.abc.x, x)

# takes in row vector and applies a function to each one
# def fn(x):
#     return np.array(list(map(f_subs, x)))

# print(np.apply_along_axis(lambda x: sortedM[5].subs(sympy.abc.x, x), axis=1, arr=np.array([[2, 3], [3, 2]])))
# print(f_subs(np.array([2, 2])))

import numpy as np
from sympy import symbols
from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key
import observables

X = np.array([
    [2, 1],
    [2, 2]
    # [2, 3],
    # [2, 4]
])

x_str = ""
for i in range(X.shape[0]):
    x_str += 'x_' + str(i) + ', '
x_syms = symbols(x_str)

M = itermonomials(x_syms, 2)
sortedM = sorted(M, key=monomial_key('grlex', np.flip(x_syms)))
print(sortedM)

# input is order up to which monomials will go
# psi = observables.monomials(3)
# print(psi.diff(X).shape)
# print(psi.diff(X)[14, 1])
# Psi_X = psi(X)

# dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)
# print(dPsiY)
# ddPsiX = psi.ddiff(X) # second order derivs