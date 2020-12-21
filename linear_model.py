# # Approach 1 - rpy2
# import pandas as pd
# from rpy2 import robjects as ro
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# R = ro.r

# x = [[1,2],[3,4]]
# y = [[3,6],[9,12]]
# df = pd.DataFrame({
#   'x': x,
#   'y': y
# })

# # B = R.lm('y~x', data=df)
# # print(B[0])
# # print(R.summary(B).rx2('coefficients')[:, 0])
# # print(B * x)


# # Approach 2 - another rpy2 that I didn't test
# import rpy2
# import statsmodels.formula.api as sm
# import pandas.rpy.common as com

# swiss = com.load_data('swiss')

# # get rid of periods in column names
# swiss.columns = [_.replace('.', '_') for _ in swiss.columns]

# # add clearly duplicative data
# swiss['z'] = swiss['Agriculture'] + swiss['Education']

# y = 'Fertility'
# x = "+".join(swiss.columns - [y])
# formula = '%s ~ %s' % (y, x)
# reg_results = sm.ols(formula, data=swiss).fit().summary()
# print(reg_results)


# Approach 3 - 1s across diagonal
import numpy as np
import observables

X = np.array([
  [1, 2],
  [2, 2],
  [2, 3],
  [2, 4]
])
m = X.shape[1]
d = X.shape[0]
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

assert np.array_equal(Bt @ Psi_X[:,0], X[:,0])
assert np.array_equal(Bt @ Psi_X[:,1], X[:,1])

# Approach 4 - mathematical expression
# import numpy as np
# import observables

# X = np.array([
#   [1, 2],
#   [2, 2],
#   [2, 3],
#   [2, 4]
# ])

# m = X.shape[1]
# psi = observables.monomials(m)
# Psi_X = psi(X)
# print(Psi_X)

# x = np.transpose(np.asmatrix(X[:,0]))

# psi_Xt = np.asmatrix(Psi_X[:,0])
# psi_X = np.transpose(psi_Xt)
# B = x @ psi_Xt @ np.linalg.pinv(psi_X @ psi_Xt)
# print(B)