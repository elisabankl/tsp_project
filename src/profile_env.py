from line_profiler import LineProfiler
from customenv import CustomEnv
import random
import numpy as np
# Create environment
env = CustomEnv(25, "Clustered")

# Create profiler
profile = LineProfiler()
profile.add_function(env.step)
profile.add_function(env._set_allowed_actions)
profile.add_function(env._get_observation)
profile.add_function(env._check_precedence_constraints)

# Run operations
env.reset()
for _ in range(10000):
    action = random.choice(np.where(env.allowed_actions))[0]
    _,_,terminated,_,_ = env.step(action)
    if terminated:
        env.reset()

# Print results
profile.print_stats()