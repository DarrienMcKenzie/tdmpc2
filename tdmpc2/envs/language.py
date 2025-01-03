import numpy as np
import gym
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random, collections

from common import math

class LanguageGenerationEnv:
    def __init__(self, prompts, tokenizer, reward_model, reward_type='dense', max_context_length=10000):
        self.prompts = prompts #unchanging original list of prompts
        self.prompt_queue = []
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_type = reward_type
        self.max_context_length = max_context_length
        self.max_episode_steps = self.max_context_length #redundant -> only added for TD-MPC2 compatability
        
        self.observation_space = gym.spaces.MultiDiscrete(self.tokenizer.vocab_size*np.ones(max_context_length))
        self.action_space = gym.spaces.Discrete(self.tokenizer.vocab_size)

        #self.prompt_queue = collections.deque(random.sample(self.prompts, len(self.prompts)))
        #self.prompt_queue = collections.deque(random.shuffle(copy.deepcopy(prompts)))
        #self.prompt_queue = copy.deepcopy(prompts) 
        #random.shuffle(self.prompt_queue)
        #self.prompt_queue = collections.deque(self.prompt_queue)


    def reset(self):
        print("*RESET*") 
        if len(self.prompt_queue) <= 1:
            self.prompt_queue = collections.deque(random.sample(self.prompts, len(self.prompts)))
        else:
            self.prompt_queue.popleft()
            
        self.true_obs = self.tokenizer(self.prompt_queue[0]).input_ids #DM-TODO: utilize batch encoding--don't encode each individually?
        self.obs = np.asarray(self.pad_obs(self.true_obs, self.max_context_length), dtype=np.float32)
        return self.obs

    def step(self, action):
        #print("*STEP*")
        action = np.argmax(action) #temporary... ideally, policy returns action instead of user needed to argmax for every env
        self.true_obs.append(action)
        self.obs = np.asarray(self.pad_obs(self.true_obs, self.max_context_length), dtype=np.float32)
        
        terminated = True if action == self.tokenizer.eos_token_id else False
        truncated = True if len(self.true_obs) >= self.max_context_length else False
        done = terminated or truncated

        if self.reward_type == 'dense':
            reward = self.reward_model(self.tokenizer.decode(self.true_obs)) #DM: skip_special_tokens=True? -> investigate
        elif self.reward_type == 'sparse':
            reward = self.reward_model(self.tokenizer.decode(self.true_obs)) if done else 0


        return self.obs, reward, done, []

    def render(self):
        print("\n*PROMPT: ", self.tokenizer.decode(self.true_obs), "\n(TOKEN VERSION): ", self.obs)
    
    def pad_obs(self, obs, max_length, padding_value=0):
        if len(obs) >= max_length:
            return obs[:max_length]  #truncate if the list is already longer
        return obs + [padding_value] * (max_length - len(obs))

    def rand_act(self):
        action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
        return math.int_to_one_hot(action, self.action_space.n)

def dummy_reward(string): #for testing purposes
    return len(string) 

def make_env(cfg):
    """
    Make LanguageGeneration environment
    """
    ds = load_dataset("openbmb/UltraFeedback")
    prompts = ds['train']['instruction']
    model = "allenai/tulu-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    context_size = 100
    reward_type = "dense"
    env = LanguageGenerationEnv(prompts, tokenizer, dummy_reward, reward_type=reward_type, max_context_length=context_size)
    return env