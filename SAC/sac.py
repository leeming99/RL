from copy import deepcopy
from gym.envs.registration import EnvRegistry, EnvSpec
from numpy.lib.polynomial import poly
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import itertools
from torch.distributions.normal import Normal


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, device):
        super().__init__()
        self.input = nn.Linear(obs_dim+act_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, obs, act):
        # obs = torch.Tensor(obs).to(self.device)
        x = F.relu(self.input(torch.cat([obs, act], dim=-1)))
        x = F.relu(self.fc1(x))
        return self.output(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit_high, act_limit_low, device):
        super().__init__()
        self.input = nn.Linear(obs_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Linear(64, act_dim)
        self.action_scale = torch.FloatTensor(
            (act_limit_high - act_limit_low) / 2.).to(device)
        self.action_bias = torch.FloatTensor(
            (act_limit_high + act_limit_low) / 2.).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.fc1(x))
        # Actor has two heads, mean and std
        mean = self.mean(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample() # sample from normal distribution using reparameterization trick
        y = torch.tanh(z) # squash output
        action = y * self.action_scale + self.action_bias # scale output

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) +  1e-6) # not sure what this does 
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


# class ActorCritic(nn.Module): 
#     # we inherit nn.Module so that we can get all param using actorcritic.parameters()
#     def __init__(self, obs_dim, act_dim, act_limit):
#         super().__init__()
#         self.pi = Actor(obs_dim, act_dim, act_limit)
#         self.q1 = QFunction(obs_dim, act_dim)
#         self.q2 = QFunction(obs_dim, act_dim)


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
            np.take(self.done, mini_batch_indices),
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
        q_lr = 1e-3,
        pi_lr = 3e-4,
        learning_starts = 5e3,
        discount_rate = 0.99,
        batch_size = 256, 
        entropy_coefficient = 0.2,
        polyak_coefficient = 0.005
    ):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = Observations(env.observation_space.shape[0], env.action_space.shape[0], int(replay_buf_size))
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low, self.device).to(self.device)
        self.q1 = QFunction(env.observation_space.shape[0], env.action_space.shape[0], self.device).to(self.device)
        self.q2 = QFunction(env.observation_space.shape[0], env.action_space.shape[0], self.device).to(self.device)
        self.q_target1, self.q_target2 = deepcopy(self.q1).to(self.device), deepcopy(self.q2).to(self.device)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.q_params, lr=q_lr)
        self.step_count = 0
        self.learning_starts = learning_starts
        self.gamma = discount_rate
        self.batch_size = batch_size
        self.alpha = entropy_coefficient
        self.tau = polyak_coefficient

    def choose_action(self, s):
        if self.step_count < self.learning_starts:
            return self.env.action_space.sample()
        action, _, _ = self.actor.get_action(torch.Tensor([s]).to(self.device).float())
        return action.tolist()[0]

    def learn(self, s, a, r, s_, done):
        self.step_count += 1
        self.replay_buffer.append(s, a, r, s_, done)

        if self.replay_buffer.curr_size > self.batch_size:
            self.update()
        return self.choose_action(s_)
        

    def update(self):
        s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        d = torch.tensor(d, device=self.device)
        s_ = torch.tensor(s_, device=self.device)

        def update_q_funct():
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.actor.get_action(s_)
                q1_targ_val = self.q_target1.forward(s_, next_actions)
                q2_targ_val = self.q_target2.forward(s_, next_actions)
                q_target = torch.min(q1_targ_val, q2_targ_val)
                target = r + self.gamma*(1-d.long())*(q_target - self.alpha*next_log_prob).view(-1)
            q1_val = self.q1.forward(s, a).view(-1) # view(-1) == flatten()
            q2_val = self.q2.forward(s, a).view(-1)
            q1_loss, q2_loss = nn.MSELoss()(q1_val, target), nn.MSELoss()(q2_val, target)
            q_loss = (q1_loss + q2_loss)/2
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

        def update_policy():
            on_policy_actions, log_prob, _ = self.actor.get_action(s)
            q1_val, q2_val = self.q1(s, on_policy_actions), self.q2(s, on_policy_actions)
            q_val = torch.min(q1_val, q2_val).view(-1)
            actor_loss = (-q_val + self.alpha*log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        def polyak_average(actor, target):
            for act, targ in zip(actor.parameters(), target.parameters()):
                targ.data = (targ.data*(1-self.tau)) + (act.data*self.tau)

        update_q_funct()
        update_policy()
        polyak_average(self.q1, self.q_target1)
        polyak_average(self.q2, self.q_target2)



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
