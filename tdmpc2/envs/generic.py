from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
import gym

def make_env(cfg):
	"""
	Make generic gym environment (ideally with discrete actions?)
	"""
	#assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	#env = gym.make('Pendulum-v1', g=9.81)
	#env = gym.make("Acrobot-v1")
	env = gym.make("CartPole-v1")
	"""
	env = gym.make(
	"LunarLander-v2",
	continuous = True,
	gravity = -10.0,
	enable_wind = False,
	wind_power = 15.0,
	turbulence_power = 1.5,
	)
	"""
	return env