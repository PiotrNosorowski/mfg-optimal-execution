from hjb import solve_hjb
from mean_field import E, dE, kappa, phi, alpha, A, T, E0
import numpy as np

t = np.linspace(0, T, 101)

# crowd's equilibrium flow
mu = dE

sol = solve_hjb(mu, kappa, phi, alpha, A, T)

# parabola coefficients at the 101 times t
h2, h1, h0 = sol.sol(t)

# right hand side of (14 C-L)
rhs = 2*kappa*dE(t) + h2*E(t)

# difference between h1 from hjb and h1 from mean field
print(np.max(np.abs(h1 - rhs)))

# parabola coefficients at t=0
h2_0, h1_0, h0_0 = sol.sol(0)

# investor at the mean vs crowd avg speed
print((h1_0 - E0*h2_0)/(2*kappa), dE(0))
