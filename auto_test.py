import numpy as np
# import auto_diff

def f(x):
  return x**3

x = np.array([[1],[3]])

# with auto_diff.AutoDiff(x) as x:
#   f_eval = f(x)
#   y, Jf = auto_diff.get_value_and_jacobians(f_eval)
#   print(y)
#   print(Jf)

# second deriv

def gradient_n(arr, n, d=1, axis=0):
    """Differentiate np.ndarray n times.

    Similar to np.diff, but additional support of pixel distance d
    and padding of the result to the same shape as arr.

    If n is even: np.diff is applied and the result is zero-padded
    If n is odd: 
        np.diff is applied n-1 times and zero-padded.
        Then gradient is applied. This ensures the right output shape.
    """
    n2 = int((n // 2) * 2)
    diff = arr

    if n2 > 0:
        a0 = max(0, axis)
        a1 = max(0, arr.ndim-axis-1)
        diff = np.diff(arr, n2, axis=axis) / d**n2
        diff = np.pad(diff, tuple([(0,0)]*a0 + [(1,1)] +[(0,0)]*a1),
                    'constant', constant_values=0)

    if n > n2:
        assert n-n2 == 1, 'n={:f}, n2={:f}'.format(n, n2)
        diff = np.gradient(diff, d, axis=axis)

    return diff

# model data from function f(x) = x^3
# second derivative should be 6x
print(gradient_n(np.array([1, 8, 27, 64, 125]), 3))