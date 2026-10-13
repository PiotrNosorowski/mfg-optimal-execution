from nash_et import count
from mean_field import E
import numpy as np
import matplotlib.pyplot as plt

Ns, gaps = [], []

for N in [2, 5, 10, 20, 50, 100]:
    # (from nash_et) what returned is: t, g0 * y(t) / y(0)
    # time grid, inventory of each player
    t_N, E_N = count(N, 1/N)

    # mean inventory of the crowd: N-player nash vs mean field limit
    gap = np.max(np.abs(E_N - E(t_N)))

    Ns.append(N)
    gaps.append(gap)

    print(f"N={N}  gap={gap:.5f}")



plt.plot(Ns, gaps, "o-", label="max |E_N(t) - E(t)|")
plt.xlabel("number of players N")
plt.ylabel("gap to mean field limit")
plt.legend()
plt.savefig("n_scaling.png", dpi=150)