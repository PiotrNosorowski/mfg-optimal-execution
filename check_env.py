from marl_env import Execution
import numpy as np
from nash_et import count

kappa = 1.0
phi = 0.25
A = 1.0
T = 10.0
g0 = 10.0
n_steps = 1000

def nash_speed(N, t):

    gamma = 1/N

    alpha = (N-1)*gamma
    A_N = A - gamma/2
    E0 = g0

    # constants of the closed-form solution of E&T (3.31) = C-L (17) 
    # alpha -> (N-1)*gamma,
    # A -> A - gamma/2
    theta = np.sqrt(kappa*phi + alpha**2/16) / kappa
    r_plus = -alpha/(4*kappa) + theta
    r_minus = -alpha/(4*kappa) - theta
    a = (alpha/4 + kappa*theta - A_N) * np.exp(-theta*T) / (
        -alpha/2*np.sinh(theta*T) + 2*kappa*theta*np.cosh(theta*T) + 2*A_N*np.sinh(theta*T))

    dE = E0*a*(r_plus*np.exp(r_plus*t) - r_minus*np.exp(r_minus*t)) + E0*r_minus*np.exp(r_minus*t)

    # since env actions are selling rates
    return -dE


def run(N):
    gamma = 1/N
    env = Execution(N, kappa, gamma, phi, T, g0, n_steps, A)
    env.reset()

    dt = T / n_steps
    positions = [g0]        # agent_0's inventory over time

    for k in range(n_steps):
        u = nash_speed(N, k*dt)
        actions = {agent: u for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        positions.append(env.positions["agent_0"])

    return positions

for N in [2, 5, 10, 20, 50, 100]:
    positions = run(N)
    t_N, E_N = count(N, 1/N)
    print(N, positions[-1], E_N[-1])