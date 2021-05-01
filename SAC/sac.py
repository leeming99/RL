from copy import deepcopy
from gym.envs.registration import EnvRegistry, EnvSpec
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import itertools


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.input = nn.Linear(obs_dim+act_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, obs, act):
        x = F.relu(self.input(torch.cat([obs, act], dim=-1)))
        x = F.relu(self.fc1(x))
        return self.output(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.input = nn.Linear(obs_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.fc1(x))
        return self.act_limit * torch.tanh(self.output(x)) # scale output according to action space


class ActorCritic(nn.Module): 
    # we inherit nn.Module so that we can get all param using actorcritic.parameters()
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, act_limit)
        self.q1 = QFunction(obs_dim, act_dim)
        self.q2 = QFunction(obs_dim, act_dim)


class Observations:
    def __init__(self, obs_dim, act_dim, buf_size):
        assert buf_size>1
        self.max_size = buf_size
        self.obs_dim = obs_dim
        self.state = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.reward = np.zeros(self.max_size, dtype=np.float32)
        self.action = np.zeros([self.max_size, act_dim], dtype=np.float32)
        self.next_state = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=np.bool)
        self.pointer = 0
        self.curr_size = 0
    
    def sample(self, size):
        assert size<=self.curr_size
        mini_batch_indices = np.random.randint(self.curr_size, size=size)
        return (
            np.take(self.state, mini_batch_indices, axis=0), 
            np.take(self.action, mini_batch_indices, axis=0), 
            np.take(self.reward, mini_batch_indices), 
            np.take(self.next_state, mini_batch_indices, axis=0),
            self.done[mini_batch_indices]
        )
    
    def append(self, state, action, reward, next_state, done):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.done[self.pointer] = done
        self.pointer = (self.pointer+1)%self.max_size
        self.curr_size = min(self.curr_size+1, self.max_size)


class Agent:
    def __init__(self, 
        env: gym.Env,
        replay_buf_size=1e6,
        ):
        self.replay_buffer = Observations(env.observation_space, env.action_space, int(replay_buf_size))
        

    def choose_action(self, s):
        pass

    def learn(self, s, a, r, s_, done):

        pass


def run_cartpole_experiment(display_plot=True, plot_name=None):
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("Pendulum-v0")
    agent = Agent(env)
    r_per_eps = []
    mean_per_100_eps = []
    solved = False
    eps = 0
    while not solved:
        eps += 1
        s = env.reset()  
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            # take action, observe s' and r
            s_, r, done, _ = env.step(a)
            # get next action
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                if len(r_per_eps)%100==0: 
                    mean = np.mean(r_per_eps[-100:])
                    mean_per_100_eps.append(mean)
                    print('Average reward for past 100 episode', mean, 'in episode', eps)
                    if mean>=-180:
                        print(f'Solved* Pendulum in episode {eps}')
                        solved = True
                break
    
    r_per_eps_x = [i+1 for i in range(len(r_per_eps))]
    r_per_eps_y = r_per_eps

    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    if plot_name:
        plt.savefig(plot_name+'.png')
    if display_plot:
        plt.show()
    return r_per_eps


if __name__ == "__main__":
    run_cartpole_experiment()
