import numpy as np
import torch


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


def test_convexity(net, x, device):
    # check convexity of the net numerically
    print("running a numerical convexity test...")
    n_trials = 100
    convexity = 0
    for trial in np.arange(n_trials):
        x1 = torch.rand(x.size()).to(device)
        x2 = torch.rand(x.size()).to(device)
        alpha = torch.rand(1).to(device)

        cvx_combo_of_input = net(alpha * x1 + (1 - alpha) * x2)
        cvx_combo_of_output = alpha * net(x1) + (1 - alpha) * net(x2)

        convexity += cvx_combo_of_input.mean() <= cvx_combo_of_output.mean()
    if convexity == n_trials:
        flag = True
        print("Passed convexity test!")
    else:
        flag = False
        print("Failed convexity test!")
    return flag
