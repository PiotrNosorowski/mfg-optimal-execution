from scipy.integrate import solve_ivp

kappa = 1
phi = 0.25
alpha = 1
T = 10
q0 = 10
A = 1.0


def fun(t, y, mu, kappa, phi, alpha):
    h2, h1, h0 = y   # as h's are the unknown time states it is solved for
    
    # how parabola changes in time
    dh2 = (h2**2 - 4*kappa*phi) / (2* kappa)                # how curvature changes
    dh1 = (h1*h2 - 2*kappa*alpha*mu(t)) / (2 * kappa)       # how tilt changes
    dh0 = (- h1**2 )/ (4*kappa)                             # how parabola's shift changes (up or down)
    return [dh2, dh1, dh0] 


def solve_hjb(mu, kappa, phi, alpha, A, T):
    sol = solve_ivp(fun, [T, 0], [2*A, 0, 0], dense_output = True, 
                    args=(mu, kappa, phi, alpha), rtol=1e-10, atol=1e-12)
    return sol

if __name__ == "__main__":

    mu = lambda t: 0.0                      # test 1: no crowd
    sol = solve_hjb(mu, kappa, phi, alpha, A, T)
    print(sol.sol(0))

    # selling shrinks inventory, so mu is negative
    mu = lambda t: -1.0                     # test 2: crowd sells at speed 1
    sol = solve_hjb(mu, kappa, phi, alpha, A, T)
    print(sol.sol(0))
