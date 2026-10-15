import torch
from marl_env import Execution
from ppo import ActorCritic

kappa = 1.0; phi = 0.25; A = 1.0; T = 10.0; g0 = 10.0; n_steps = 50

N = 2

# load the trained network
model = ActorCritic()
model.load_state_dict(torch.load(f"agent_0_N{N}.pt"))

env = Execution(N, kappa, 1/N, phi, T, g0, n_steps, A)
observations, infos = env.reset()

total = 0.0
positions = [g0]

for k in range(n_steps):
    actions = {}
    for agent in env.agents:
        with torch.no_grad():
            mu, value = model.forward(torch.tensor(observations[agent], dtype=torch.float32))
        actions[agent] = mu.item()       # learned speed, no noise
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total += rewards["agent_0"]
    positions.append(env.positions["agent_0"])

print("total reward:", total)
print("position at t=1, 2, 5:", positions[5], positions[10], positions[25])