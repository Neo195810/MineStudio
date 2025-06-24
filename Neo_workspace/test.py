from minestudio.simulator import MinecraftSim
import time

sim = MinecraftSim(action_type="env")
time.sleep(1)
obs, info = sim.reset()
for _ in range(100):
    action = sim.action_space.sample()
    obs, reward, terminated, truncated, info = sim.step(action)
sim.close()