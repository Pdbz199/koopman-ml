import numpy as np
import observables
import pandas as pd

def gEDMD(X, psi):
  Psi_X = psi(X)
  nablaPsi = psi.diff(X)
  nabla2Psi = psi.ddiff(X)
  k = Psi_X.shape[0]

  # t = 1 is a placeholder time step, not really sure what it should be
  def dpsi(k, l, t=1):
    term_1 = (1/t)*(X[:, l+1]-X[:, l])
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * ((X[:, l+1]-X[:, l]) * np.transpose(X[:, l+1]-X[:, l]))
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1,term_2) + np.sum(np.multiply(term_3,term_4))

  dPsi_X = np.zeros((k, m))
  for row in range(k):
    for column in range(m-1):
      dPsi_X[row, column] = dpsi(row, column)

  # calculate \widehat{L}^\top
  M = dPsi_X @ np.linalg.pinv(Psi_X)
  # estimate of Koopman generator
  L = np.transpose(M)
  return L


X = np.array([
  [1, 2],
  [2, 2],
  [2, 3],
  [2, 4]
])

# get dataframe from CSV file
# df = (pd.read_csv('../ExportedCoinData.csv')
#       .groupby(['datetime']))

d = X.shape[0]
m = X.shape[1]
psi = observables.monomials(8)
Psi_X = psi(X)
k = Psi_X.shape[0]
Bt = np.zeros((d, k))
if Bt.shape[1] == 1:
  Bt[0,0] = 1
else:
    row = 0
    for i in range(1, d+1):
      Bt[row,i] = 1
      row += 1
B = np.transpose(Bt)

L = gEDMD(X, psi)
# print(L)

def b(l):
  return np.transpose(L @ B) * Psi_X[:, l]

# sigma = np.zeros((d, d))
# for i in range(d):
#   for j in range(d):
#     x_i = X[i]
#     x_j = X[j]
#       # a[i,j] is a function that takes in the index of \psi and x and completes the calculation
#     a[i, j] = lambda k, l: (L * Psi_X[k, l]) - (b(l)[i]*x_j) - (b(l)[j]*x_i)

# def sigma(l):
#   return (L * Psi_X[k, l]) - b(2)

# num squared or multiplied terms = (dC2) + d

'''
NOTES:

Thing  to do is
Fit model predictive value would be the mean?
MLE?
'''