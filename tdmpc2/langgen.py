import numpy as np
import gym
import torch
from transformers import AutoTokenizer#, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import random, collections
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Language Environment
class LanguageGenerationEnv:
    def __init__(self, prompts, tokenizer, reward_model, reward_tokenizer, reward_type='dense', max_context_length=10000):
        self.prompts = prompts #unchanging original list of prompts
        self.prompt_queue = []
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_type = reward_type
        self.max_context_length = max_context_length
        self.max_episode_steps = self.max_context_length #redundant -> only added for TD-MPC2 compatability
        
        self.observation_space = gym.spaces.MultiDiscrete(self.tokenizer.vocab_size*np.ones(max_context_length))
        self.action_space = gym.spaces.Discrete(self.tokenizer.vocab_size)

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
        action = np.argmax(action) #DM: temporary... ideally, policy returns action instead of user needed to argmax for every env
        self.true_obs.append(action)
        self.obs = np.asarray(self.pad_obs(self.true_obs, self.max_context_length), dtype=np.float32)
        
        terminated = True if action == self.tokenizer.eos_token_id else False
        truncated = True if len(self.true_obs) >= self.max_context_length else False
        done = terminated or truncated

        if self.reward_type == 'dense':
            reward = self.reward_model.generate(self.reward_tokenizer(self.tokenizer.decode(self.true_obs)).input_ids) #DM: skip_special_tokens=True? -> investigate
        elif self.reward_type == 'sparse':
            reward = self.reward_model.generate(self.reward_tokenizer(self.tokenizer.decode(self.true_obs)).input_ids) if done else 0


        return self.obs, reward, done, []

    def render(self):
        print("\n*PROMPT: ", self.tokenizer.decode(self.true_obs), "\n(TOKEN VERSION): ", self.obs)
    
    def pad_obs(self, obs, max_length, padding_value=0):
        if len(obs) >= max_length:
            return obs[:max_length]  #truncate if the list is already longer
        return obs + [padding_value] * (max_length - len(obs))

def dummy_reward(string): #for testing purposes
    return len(string)


# Tests:
def env_test():
    ds = load_dataset("openbmb/UltraFeedback")
    prompts = ds['train']['instruction']
    model = "allenai/tulu-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    context_size = 5000
    env = LanguageGenerationEnv(prompts, tokenizer, dummy_reward, max_context_length=context_size)

    #set_trace()

    obs = env.reset()
    while True: #training loop
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = int(input("ACTION: "))
            obs, reward, done, info = env.step(action)
            print("REWARD: ", reward)
            #while True #per-prompt loop (episode)
         
def preference_test():
    print("Loading dataset...")
    ds = load_dataset("openbmb/UltraFeedback")
    print("Dataset loaded.")
    #set_trace()
    prompts = ds['train']['instruction']
    for i, prompt in enumerate(prompts):
        print("\nPROMPT #" + str(i) + ": \n" + prompt)
        print("\nTOKENIZATION: ", tokenizer(prompt, return_tensors="pt").input_ids)
        print("\n\n")

        if i == 1000:
            print("***STOP***")
            break

def llm_inference_test():
    #device = torch.device('cuda')
    model_path = "allenai/tulu-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("Model loaded.")
    #model.to(device)
    prompt = "Who was George Washington?"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print("Peforming inference...")
    # Generate response
    output_ids = model.generate(
    input_ids,
    max_length=50,  # Adjust max_length as needed
    temperature=0.7,  # Adjust temperature for creativity
    top_p=0.9,       # Nucleus sampling for diverse outputs
    do_sample=True   # Enables sampling instead of greedy decoding
    )
    print("Inference complete")
    print("COMPLETED TEXT:\n")
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)


def eval_test():
    ds = load_dataset("openbmb/UltraFeedback")
    prompts = ds['train']['instruction']

    model = "allenai/tulu-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    reward_model = "openbmb/UltraRM-13b"
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model, use_fast=False)

    context_size = 2048
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        #print("\nPROMPT #" + str(i) + ": \n" + prompt)
        output_ids = model.generate(
        input_ids,
        max_length=50,  # Adjust max_length as needed
        temperature=0.7,  # Adjust temperature for creativity
        top_p=0.9,       # Nucleus sampling for diverse outputs
        do_sample=True   # Enables sampling instead of greedy decoding
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        complete_response = "Prompt: {prompt}\nResponse: {output_text}".format(prompt, output_text)
        #reward1 = reward_tokenizer(complete_response, return_tensors="pt").input_ids
        reward = reward_model(reward_tokenizer(complete_response, return_tensors="pt").input_ids).logits

        print("\n\nPROMPT #" + str(i) + ": " + prompt)
        print("\n\nRESPONSE: ", output_text)
        print("\n\nREWARD: ", reward)
        #print("\nTOKENIZATION: ", tokenizer(prompt, return_tensors="pt").input_ids)
        #print("\n\n")

        if i == 10:
            print("MAX OF " + str(i) + "REACHED")
            break


def main():
    print("~~START MAIN~~")
    #llm_inference_test()
    #preference_test()
    #env_test()
    eval_test()
    print("~~END MAIN~~")

if __name__ == "__main__":
    main()