import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.world_model_discrete import WorldModelDiscrete
from tensordict import TensorDict
from ipdb import set_trace

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device) if not cfg.action_mode == 'discrete' else WorldModelDiscrete(cfg).to(self.device)
		self.optim = torch.optim.Adam([
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.parameters()},
				{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
				}
			], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('compiling - update')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
		else:
			z = self.model.encode(obs, task)
			if self.cfg.critic_only: #DQN-type approach
				a = self.model.Q(z, task, 'avg', target=False, detach=False).argmax().unsqueeze(0)
			else: #Actor-critic/SAC approach
				a = self.model.pi(z, task)[0].argmax().unsqueeze(0)
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return pi_loss.detach(), pi_grad_norm
	
	def update_pi_discrete(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		actions, action_probs, log_probs = self.model.pi(zs, task)

		qs = self.model.Q(zs, task, return_type='avg', detach=True)
		#self.scale.update(qs[0].gather(1,actions[0].squeeze().argmax(1).unsqueeze(1)))
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))

		#DM: both losses are monotonically increasing...

		#DM: Yutao's loss
		pi_loss = ((action_probs*((self.cfg.entropy_coef * log_probs) - qs)).mean(dim=(1,2)) * rho).mean() 

		"""
		#DM: Modified loss, decomposing into terms for debugging purposes
		entropy_term = (self.cfg.entropy_coef * log_probs)#[0]#no horizon test #.gather(2, actions)
		value_term = qs#[0]#no horizon test#.gather(2, actions)
		entropy_value_diff_term = entropy_term - value_term
		action_prob_term = action_probs.transpose(1,2)#[0] #no horizon test
		exact_expectation = torch.bmm(action_prob_term,entropy_value_diff_term) #performing batched matrix multiplication
		pi_loss = (exact_expectation.sum((1,2))*rho).mean()
		"""

		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return pi_loss.detach(), pi_grad_norm

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1] if self.cfg.get('action_mode') == 'discrete' else self.model.pi(next_z, task)[0]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

	def _update(self, obs, action, reward, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_loss, pi_grad_norm = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"pi_loss": pi_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		}).detach().mean()
	
	def _td_target_discrete(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		
		if self.cfg.early_stopping:
			terminal = (self.model.termination(next_z, task) >= 0.50).to(torch.float32) 
		else:
			terminal = 0

		#MODIFIED (version in which TD targets are only calculated/updated for the action that was taken)
		if not self.cfg.critic_only: #testing with SAC elements
			#ORIGINAL (from Yutao's branch):
			next_actions, next_act_prob, next_log_prob = self.model.pi(next_z, task)
			next_q_target = self.model.Q(next_z, task, return_type='min', target=True)
			min_q_next_target = next_act_prob * (next_q_target - (self.cfg.entropy_coef * next_log_prob))
			min_q_next_target = min_q_next_target.sum(dim=2, keepdim=True)
		else: #DM: an ablation; testing without SAC elements (CRITIC/VALUE ONLY)
			Qz = self.model.Q(next_z, task, return_type='min', target=True) #DM-POI: Qs are the same for every 256?
			min_q_next_target = Qz.max(2).values.unsqueeze(2)
			
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		td_targets = reward + discount * min_q_next_target * (1-terminal)
		return td_targets

	def _update_discrete(self, obs, action, reward, done=None, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target_discrete(next_z, reward, task)

		# Encode actions
		action = F.one_hot(action.squeeze().long(),num_classes=self.cfg.action_dim)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss, terminal_loss = 0, 0
		#for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0), done.unbind(0))):
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			#DM-POI: termination needs to inform consistency...
			z = self.model.next(z, _action, task)
			#terminal_pred = self.model.termination(_zs, task) 

			#if termination_preds > 50%, then don't calculate consistency loss on remaining zs...
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			#terminal_loss = termination_loss + F.binary_cross_entropy(terminal_pred.to(torch.float32), done.to(torch.float32)) * self.cfg.rho**t
			
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, termination_loss, value_loss = 0, 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				### NON-ENCODED CASES:

				## Original (Yutao's) version:
				#qs_unbind_unbind_act = qs_unbind_unbind.gather(1,action[t].long()).view(-1)
				#value_loss = value_loss + torch.nn.functional.mse_loss(qs_unbind_unbind_act, td_targets_unbind).mean() * self.cfg.rho**t

				## Modified version (relies on modified td_targets calculations):
				#value_loss = value_loss + torch.nn.functional.mse_loss(qs_unbind_unbind.gather(1,action[t].long()).view(-1), td_targets_unbind.view(-1), self.cfg) * self.cfg.rho**t

				### ONE-HOT ENCODED CASE:
				sampled_actions = action[t].argmax(dim=1).unsqueeze(1)
				value_loss = value_loss + F.mse_loss(qs_unbind_unbind.gather(1, sampled_actions), td_targets_unbind, self.cfg) * self.cfg.rho**t
				

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		terminal_loss = terminal_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.terminal_coef * terminal_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss 
		)


		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		if not self.cfg.critic_only:
			pi_loss, pi_grad_norm = self.update_pi_discrete(zs.detach(), task)

			# Update target Q-functions
			self.model.soft_update_target_Q()

			# Return training statistics
			self.model.eval()
			return TensorDict({
				"consistency_loss": consistency_loss,
				"reward_loss": reward_loss,
				"value_loss": value_loss,
				"pi_loss": pi_loss,
				"total_loss": total_loss,
				"grad_norm": grad_norm,
				"pi_grad_norm": pi_grad_norm,
				"pi_scale": self.scale.value,
			}).detach().mean()
		else:
			# Update target Q-functions
			self.model.soft_update_target_Q()

			# Return training statistics
			self.model.eval()
			return TensorDict({
				"consistency_loss": consistency_loss,
				"reward_loss": reward_loss,
				"value_loss": value_loss,
				"total_loss": total_loss,
				"grad_norm": grad_norm,
			}).detach().mean()
	
	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, done, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, **kwargs) if self.cfg.get('action_mode') != 'discrete' else self._update_discrete(obs, action, reward, done, **kwargs)
