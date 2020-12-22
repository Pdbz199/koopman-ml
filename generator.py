# This code is based on gEDMD theory from Klus et al. (https://arxiv.org/pdf/1909.10638.pdf)
# Throughout, l (letter ell) will be used to refer to x_l, or the lth snapshot of the state observables, X

import math
import numpy as np
import observables
import pandas as pd
from sympy import symbols
from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key

# Construct B matrix as seen in 3.1.2 of the reference paper
def constructB(d, n):
    Bt = np.zeros((d, n))
    if Bt.shape[1] == 1:
        Bt[0,0] = 1
    else:
        row = 0
        for i in range(1, d+1):
            Bt[row,i] = 1
            row += 1
        B = np.transpose(Bt)
    return B

# Construct similar B matrix as above, but for second order monomials
def constructSecondOrderB(s, n):
    Bt = np.zeros((s, n))
    if Bt.shape[1] == 1:
        Bt[0,0] = 1
    else:
        row = 0
        for i in range(d+1, d+1+s):
            Bt[row,i] = 1
            row += 1
    B = np.transpose(Bt)
    return B

# Get dataframe from CSV file
coin_data = pd.read_csv('coindatav4.csv')
coin_data = coin_data.drop(columns=['Unnamed: 0', 'coin', 'price']) # returns
g = coin_data.groupby('datetime').cumcount()
X = np.transpose(np.array((coin_data.set_index(['datetime',g])
        .unstack(fill_value=0)
        .stack().groupby(level=0)
        .apply(lambda x: np.array(x.values.tolist()).reshape(len(x)))
        .tolist())))
X = X[:, int(X.shape[1]*0.80):] # last 20% of snapshots

# Various setup variables and definitions
d = X.shape[0]
m = X.shape[1]
s = int(d*(d+1)/2) # number of second order poly terms
psi = observables.monomials(4) # I don't have enough memory for 5+
Psi_X = psi(X)
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)
n = Psi_X.shape[0]

# This computes dpsi_k(x) exactly as in the paper
# t = 1 is a placeholder time step, not really sure what it should be
def dpsi(k, l, t=1):
    term_1 = (1/t) * ((X[:, l+1]-X[:, l]).reshape(1, -1))
    term_2 = nablaPsi[k, :, l].reshape(-1, 1)
    term_3 = (1/(2*t)) * ((X[:, l+1]-X[:, l]).reshape(-1, 1) * (X[:, l+1]-X[:, l]).reshape(1, -1))
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1,term_2) + np.sum(term_3@term_4)

# Construct \text{d}\Psi_X matrix
dPsi_X = np.zeros((n, m))
for row in range(n):
    for column in range(m-1):
        dPsi_X[row, column] = dpsi(row, column, d)

# Calculate Koopman generator approximation
train = int(m * 0.8)
test = m - train
M = dPsi_X[:, :train] @ np.linalg.pinv(Psi_X[:, :train]) # \widehat{L}^\top
L = M.T # estimate of Koopman generator

# Construct B matrix (selects first-order monomials except 1)
B = constructB(d, n)

# Computed b function (sometimes called \mu)
def b(l):
    return (L @ B).T @ Psi_X[:, l].reshape(-1, 1)

# Eigen decomposition
eig_vals, eig_vecs = np.linalg.eigh(L)
# Calculate Koopman modes
V = np.transpose(B) @ np.linalg.inv((eig_vecs).T)
# Compute eigenfunction matrix
eig_funcs = (eig_vecs).T @ Psi_X

# This b function allows for heavy dimension reduction!
# default is reducing by 90% (taking the first n/10 eigen-parts)
def b_v2(l, num_dims=n//10):
    res = 0
    for ell in range(n-1, n-num_dims, -1):
        res += eig_vals[ell].reshape(-1, 1) * eig_funcs[ell, l].reshape(-1, 1) * V[:, ell].reshape(-1, 1)
    return res

# Construct second order B matrix (selects second-order monomials)
second_orderB = constructSecondOrderB(s, n)

# the a function
# this was calculated in a weird way, so could have issues...
def a(l):
    return (np.transpose(L @ second_orderB) @ Psi_X[:, l].reshape(-1, 1)) - ((np.transpose(second_orderB) @ nablaPsi[:, :, l]) @ b_v2(l))

x_str = ""
for i in range(d):
    x_str += 'x_' + str(i) + ', '
x_syms = symbols(x_str)
M = itermonomials(x_syms, 4)
sortedM = sorted(M, key=monomial_key('grlex', np.flip(x_syms)))

# Function to compute the a matrix at a specific snapshot index
def evalAMatrix(l):
    a_matrix = np.zeros((d, d))
    for p in range(d+1, d+1+s):
        monomial = str(sortedM[p])
        i = 0
        j = 0
        split_mon = monomial.split('**')
        if len(split_mon) > 1:
            i = int(split_mon[0][-1])
            j = int(split_mon[0][-1])
        else:
            split_mon = monomial.split('*')
            i = int(split_mon[0][-1])
            j = int(split_mon[1][-1])

        a_matrix[i,j] = a(l)[p-d-1]
        a_matrix[j,i] = a(l)[p-d-1]

    return a_matrix

# Oh no... it's not positive definite
# Some calculation must be wrong, darn...
# decomp = np.linalg.cholesky(evalAMatrix(0))

def sigma(l):
    # Attempt at work around without Cholesky
    U, S, V = np.linalg.svd(evalAMatrix(l))
    square_S = np.diag(S**(1/2))
    sigma = V @ square_S @ V.T
    return sigma

def epsilon_t(l):
    np.linalg.inv(sigma(l-1).T @ sigma(l-1)) @ sigma(l-1).T @ (X[:, l].reshape(-1, 1) - b_v2(l-1))

# snapshots by coins
# rows are snapshots
epsilons = np.zeros((m, d))
for snapshot_index in range(1, m):
    epsilons[snapshot_index] = epsilon_t(snapshot_index)

# Epsilons produced make no sense...
# We are looking for numbers that follow a
# normal distribution but these are way off

print(epsilons.shape)
epsilons = epsilons.T
print(epsilons.shape)
# np.save('saved_epsilons', epsilons)