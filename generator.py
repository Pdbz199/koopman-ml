import numpy as np
# import auto_diff
import observables

X = np.array([
  [1, 2],
  [2, 2],
  [2, 3],
  [2, 4]
])

psi = observables.monomials(3)
Psi_X = psi(X)
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)
k = nablaPsi.shape[0]
m = nablaPsi.shape[1]

t = 1 # placeholder time step
def dpsi(k, l):
  term_1 = (1/t)*(X[l+1]-X[l])
  term_2 = nablaPsi[k,l]
  term_3 = (1/(2*t)) * ((X[l+1]-X[l]) * np.transpose(X[l+1]-X[l]))
  term_4 = nabla2Psi[k,l]
  return np.dot(term_1,term_2) + np.multiply(term_3, term_4)

dPsi_X = np.zeros((k, m)) # ?
for row in range(k):
  for column in range(m):
    print(dpsi(row,column))
    print(dpsi(row,column).shape)
    dPsi_X[row,column] = dpsi(row,column)

# calculate \hat{L}^\top
# M = dPsi_X * pseudoinverse(Psi_X)
# calculate Koopman generator estimator
# L = t(M)