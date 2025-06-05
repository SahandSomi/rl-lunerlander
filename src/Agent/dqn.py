import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim


class DQN():
    """Interacts with and learns from the environment."""

    def __init__(self, Network, state_size, action_size , ReplyBuffer, device, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(config.SEED)
        self.device = device

        # Q-Network
        self.qnetwork_local = Network.to(self.device)
        self.qnetwork_target = Network.to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.LR)

        # Replay memory
        self.memory = ReplyBuffer
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
                # For reward normalization
        self.reward_buffer = []
        self.reward_mean = 0
        self.reward_std = 1
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Update reward statistics for normalization
        self.reward_buffer.append(reward)
        self.min_reward = min(self.reward_buffer)
        self.max_reward = max(self.reward_buffer)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

    def normalize_reward(self, reward):
        """Normalize reward to be between -1 and 1 using normal distribution statistics."""
        normalized_reward = (2*(reward-(self.min_reward))/((self.max_reward-self.min_reward)))-1 # Assuming rewards are in the range [-1000, 200] x′′=2x−minxmaxx−minx−1
        return normalized_reward

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #normalized_rewards = self.normalize_reward(rewards)
        normalized_rewards = rewards

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = normalized_rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def behavior_cloning(self, state, action):
        """Behavior cloning method to train the agent."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = torch.tensor(action).unsqueeze(0).to(self.device)

        self.qnetwork_local.train()
        self.optimizer.zero_grad()

        # Forward pass
        action_values = self.qnetwork_local(state)
        loss = F.mse_loss(action_values, action.float())

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()