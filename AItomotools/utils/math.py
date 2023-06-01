import numpy as np

##TODO : Type hinting, what does this function returns? def func(args) -> Type:
def power_method(op, maxiter=100, tol=1e-6):
    arr_old = np.random.rand(*op.domain_shape).astype(np.float32)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = np.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.T(u)
        beta = np.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = np.linalg.norm(op(arr) - beta * u)
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma
