from collections import defaultdict

import gym
import numpy as np
import torch

class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env, cfg=None):
		super().__init__(env)
		self.cfg = cfg
	
	def rand_act(self):
		if self.cfg.action_mode == 'discrete':
			#generate random distribution over actions number
			act_probs = np.random.rand(self.action_space.n)
			return torch.tensor(act_probs.astype(np.float32))

		sample = [self.action_space.sample()]
		sample = np.array(sample)
		# Convert to tensor
		if sample.ndim == 2:
			sample = sample[0]

		return torch.from_numpy(sample.astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		if self.cfg.action_mode == "discrete":
			#action = int(action.argmax())
			#BEFORE ENC_ACTION CHANGE
			if not self.cfg.critic_only:
				action = int(action[0])
			else:
				action = int(action)
		else:
			action = action.numpy()
	
		obs, reward, done, info = self.env.step(action)
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info