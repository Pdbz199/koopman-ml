import math
import numpy as np
import observables
import pandas as pd

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

def constructSecondOrderB(d, n):
  new_dim = int(d*(d+1)/2)
  Bt = np.zeros((new_dim, n))
  if Bt.shape[1] == 1:
    Bt[0,0] = 1
  else:
      row = 0
      for i in range(d+1, d+1+new_dim):
        Bt[row,i] = 1
        row += 1
  B = np.transpose(Bt)
  return B

# Dataset
X = np.array([
  [1, 2, 3],
  [2, 3, 4],
  [2, 3, 4],
  [2, 3, 4]
])

# get dataframe from CSV file
# code takes like 5-10 minutes to run
# coin_data = pd.read_csv('../coindatav4.csv')
# print(coin_data.columns)
# coin_data = coin_data.drop(columns=['Unnamed: 0', 'coin', 'returns'])

# g = coin_data.groupby('datetime').cumcount()
# X = np.transpose(np.array((coin_data.set_index(['datetime',g])
#         .unstack(fill_value=0)
#         .stack().groupby(level=0)
#         .apply(lambda x: np.array(x.values.tolist()).reshape(len(x)))
#         .tolist())))
# print(X.shape)
d = X.shape[0]
m = X.shape[1]
psi = observables.monomials(4) # I don't have enough memory for 5+
Psi_X = psi(X)
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)
n = Psi_X.shape[0]

# t = 1 is a placeholder time step, not really sure what it should be
def dpsi(k, l, t=1):
  term_1 = (1/t)*(X[:, l+1]-X[:, l])
  term_2 = nablaPsi[k, :, l]
  term_3 = (1/(2*t)) * ((X[:, l+1]-X[:, l]) * np.transpose(X[:, l+1]-X[:, l]))
  term_4 = nabla2Psi[k, :, :, l]
  return np.dot(term_1,term_2) + np.sum(np.multiply(term_3,term_4))

# Construct \text{d}\Psi_X matrix
dPsi_X = np.zeros((n, m))
for row in range(n):
  for column in range(m-1):
    dPsi_X[row, column] = dpsi(row, column)

# Calculate Koopman generator approximation
M = dPsi_X @ np.linalg.pinv(Psi_X) # \widehat{L}^\top
L = np.transpose(M) # estimate of Koopman generator
# print(L.shape)

# Eigen decomposition
eig_vals, eig_vecs = np.linalg.eigh(L)
# print(eig_vals.shape)
# print(eig_vecs.shape)

# Construct B matrix (selects first-order monomials except 1)
B = constructB(d, n)
# Calculate Koopman modes
V = np.transpose(B) @ np.linalg.inv(np.transpose(eig_vecs))
print(V.shape)
# Compute eigenfunctions
eig_funcs = np.transpose(eig_vecs) @ Psi_X
# print(eig_funcs)

def bb(l):
  # DIMENSION REDUCTIONNnNnnnnNNNNNNNNn
  num_dims = 8
  res = 0
  for ell in range(n-1, n-num_dims, -1):
    res += eig_vals[ell] * eig_funcs[ell, l] * V[:, ell]
  return res

print(bb(0))

# b function
def b(l):
  return np.transpose(L @ B) @ Psi_X[:, l]

print(b(0))

# Construct second order B matrix (selects second-order monomials)
second_orderB = constructSecondOrderB(d, n)

# a function
def a(l):
  return (np.transpose(L @ second_orderB) @ Psi_X[:, l]) - ((np.transpose(second_orderB) @ nablaPsi[:, :, l]) @ b(l))



'''
NOTES:

Thing  to do is
Fit model predictive value would be the mean?
MLE?
'''