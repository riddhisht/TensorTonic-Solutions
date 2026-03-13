def gradient_descent_quadratic(a, b, c, x0, lr, steps):

    for i in range(steps):

        fbar = 2*a*x0 + b
        x0 = x0 - lr *fbar

    return x0
        