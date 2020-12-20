import numpy as np
import observables

def gEDMD(X):
  m = X.shape[0]
  psi = observables.monomials(m)
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
      dPsi_X[row,column] = dpsi(row,column)

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

L = gEDMD(X)
print(L)