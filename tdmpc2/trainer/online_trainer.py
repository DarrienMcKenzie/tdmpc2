from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from ipdb import set_trace

DISCRETE = True
class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				#action = self.agent.act(obs, t0=t==0, eval_mode=True) #ORIGINAL
				action = self.env.rand_act() #DM-TEMP
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			print("ORIGINAL OBS IS DICT")
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		
		if action is None:
			#DM-MODIFIED
			if not DISCRETE:
				action = torch.full_like(self.env.rand_act(), float('nan')) #ORIGINAL
			else:
				action = torch.Tensor((self.env.rand_act(),))
				
		if reward is None:
			reward = torch.tensor(float('nan'))
			
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0), #ORIGINAL
			reward=reward.unsqueeze(0),
			batch_size=(1,)) #ORIGINAL BATCH_SIZE=(1,)
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				#print("DELIBERATE ACTION")
				action = self.agent.act(obs, t0=len(self._tds)==1) #ORIGINAL
				#action = torch.argmax(torch.nn.functional.softmax(action))
				#set_trace()
			else:
				action = self.env.rand_act()
				#print("RANDOM ACTION")
				#if DISCRETE:
				#	action = torch.tensor((action,))
			obs, reward, done, info = self.env.step(int(action)) #DM: brute force test...

			#DM-MOD
			if DISCRETE:
				action = torch.tensor((action,))
			#DM-MODIFIED

			self._tds.append(self.to_td(obs, action, reward)) #ORIGINAL #DM: BRUTE FORCE TESTING

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				#DM: Policy update entry point
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)