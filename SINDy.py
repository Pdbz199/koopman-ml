import numpy as np
import observables

def SINDy(X, eps=0.001, iterations=10):
    m = X.shape[0]
    psi = observables.monomials(m)
    PsiX = psi(X)

    # not sure what this is supposed to be
    Xi = Y @ _sp.linalg.pinv(PsiX) # least-squares initial guess

    for k in range(iterations):
        s = abs(Xi) < eps # find coefficients less than eps ...
        Xi[s] = 0         # ... and set them to zero

        for ind in range(m): # for each snapshot
            b = ~s[ind, :] # consider only functions corresponding to coefficients greater than eps
                           # '~' operator flips T/F of s
                           # if arr = [1,2] then arr[[False, True]] == [2]
            Xi[ind, b] = Y[ind, :] @ _sp.linalg.pinv(PsiX[b, :])
    return Xi