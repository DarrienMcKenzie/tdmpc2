
from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')
try:
	from envs.simple_env import make_env as make_classic_env
except:
	make_classic_env = missing_dependencies
try:
	from envs.atari import make_atari as make_atari_env
except:
	make_atari_env = missing_dependencies
try:
	from envs.simple_env import make_simple as make_simple_env
except:
	make_simple_env = missing_dependencies
try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		env = None
		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_atari_env, make_classic_env]:
			try:
				env = fn(cfg)
			except ValueError:
				pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env,cfg)
	if cfg.get('obs', 'state') == 'rgb' and cfg.task_platform != 'atari':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	try:
		cfg.action_dim = env.action_space.shape[0]
	except:
		if cfg.action_mode == 'discrete':
			cfg.action_dim = env.action_space.n
		else:
			cfg.action_dim = 1
			cfg.action_range = env.action_space.n #for atari discrete action space, naively output action index

	cfg.episode_length = env.max_episode_steps if hasattr(env, 'max_episode_steps') else cfg.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	# cfg.seed_steps = 200
	return env

