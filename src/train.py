import random
from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.96,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000, 
          'epsilon_delay_decay': 100,
          'batch_size': 800,
          'gradient_steps': 5,
          'update_target_strategy': 'replace',
          'update_target_freq': 1000,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()}

class ProjectAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = self.Mymodel()
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']

    def act(self, observation, use_random=False):
      if use_random:
        return env.action_space.sample()
      else:
        return self.greedy_action(self.model, observation)

    def save(self):
        self.path = os.getcwd() + "/model.pth"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/model.pth"
        self.model = self.Mymodel()
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return

    def greedy_action(self, network, state): #Jenna
      device = "cuda" if next(network.parameters()).is_cuda else "cpu"
      with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def Mymodel(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons=256
        model = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)).to(self.device)
        return model

    def train(self):
        previous_val = 0
        max_episode = 400 
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            action=self.act(state,np.random.rand()< epsilon)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            if self.update_target_strategy == 'replace': 
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            step += 1
            if done or trunc:
                episode += 1
                validation_score = evaluate_HIV(agent=self, nb_episode=5)
                print("Episode ", '{:3d}'.format(episode), 
                  ", epsilon ", '{:6.2f}'.format(epsilon), 
                  ", batch size ", '{:5d}'.format(len(self.memory)), 
                  ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                  ", score ", '{:4.1f}'.format(validation_score),
                  sep='')
                state, _ = env.reset()
                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(self.device)
                    self.save()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        self.model.load_state_dict(self.best_model.state_dict())
        self.save()
        return episode_return

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity 
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
