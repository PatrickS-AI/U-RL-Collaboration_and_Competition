import numpy as np
import random
import copy
from collections import namedtuple, deque

from maddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process to generate noise for action selection."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.19):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size,n_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.n_agents = n_agents
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # There is one state and action per agent, but one reward and done for each timestep
        l_states = [torch.from_numpy(np.vstack([e.state[idx] for e in experiences if e is not None])).float().to(device) for idx in range(self.n_agents)]
        l_actions = [torch.from_numpy(np.vstack([e.action[idx] for e in experiences if e is not None])).float().to(device) for idx in range(self.n_agents)]
        l_next_states = [torch.from_numpy(np.vstack([e.next_state[idx] for e in experiences if e is not None])).float().to(device) for idx in range(self.n_agents)]            
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (l_states, l_actions, rewards, l_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



            
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,n_agents, buffer, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        # Set given state and action sizes
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network having local and target network for soft updates
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network having local and target network for soft updates
        self.critic_local = Critic(state_size, action_size,n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size,n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process to boost exploration and hence learning of the network
        self.noise = OUNoise(action_size, random_seed)
        
        self.memory = buffer
        
            
    def step(self):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        if len(self.memory) > BATCH_SIZE: # Do only if batch is full 
            experiences = self.memory.sample() # draw sample
            self.learn(experiences, GAMMA)        

    def act(self, state, add_noise=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device) # from numpy state to torch tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() # Forward state to get action probs
        self.actor_local.train()
        action += self.noise.sample()*add_noise
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        l_states, l_actions, rewards, l_next_states, dones = experiences  
        t_states      = torch.cat(l_states, dim=1).to(device)
        t_actions     = torch.cat(l_actions, dim=1).to(device)
        t_next_states = torch.cat(l_next_states, dim=1).to(device)    
        
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        t_next_actions = torch.cat([self.actor_target(states) for states in l_states] , dim=1).to(device)        
        Q_targets_next = self.critic_target(t_next_states, t_next_actions)        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))        
        # Compute critic loss
        Q_expected = self.critic_local(t_states, t_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions        
        t_actions_pred = torch.cat([self.actor_local(states) for states in l_states] , dim=1).to(device)
        actor_loss = -self.critic_local(t_states, t_actions_pred).mean()        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()        
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            

class MADDPG:
    """ Multi Agent class that creates and manages a defined amount of agents together with their their shared buffer"""

    def __init__(self,state_size,action_size,n_agents, random_seed):
        """Initialize an MADDPG object.
        Params
            n_agents    (int): number of ddpg agents
            random_seed (int): random seed
        """
        self.n_agents = n_agents
        self.action_size = action_size
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE,n_agents)
        self.agents = [Agent(state_size,action_size,n_agents,self.memory,random_seed) for x in range(n_agents)]
        
    def step(self, states, actions, rewards, next_states, dones):
        """Adds memory to shared buffer and executes step for each agent."""
        self.memory.add(states, actions, rewards, next_states, dones)
        for agent in self.agents:
            agent.step()

    def act(self, states, noise = 1.0):
        """Get action of all available agents"""
        actions = np.zeros([self.n_agents, self.action_size])
        for i in range(self.n_agents):
            actions[i, :] = self.agents[i].act(states[i], noise)
        return actions

    def store_agents(self):
        """Stores actor and critic weights of all agents"""
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agt{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agt{}_checkpoint_critic.pth'.format(index+1))
    
    def reset(self):
        """ Reset all agents"""
        for agent in self.agents:
            agent.reset()
  