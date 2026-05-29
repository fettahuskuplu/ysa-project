import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import GRUDQNConfig
from gru_model import GRUDQN
from replay_buffer import ReplayBuffer


class GRUDQNAgent:
    def __init__(self, input_size: int):
        self.epsilon = GRUDQNConfig.EPSILON_START
        self.epsilon_end = GRUDQNConfig.EPSILON_END
        self.epsilon_decay = GRUDQNConfig.EPSILON_DECAY
        self.gamma = GRUDQNConfig.GAMMA
        self.batch_size = GRUDQNConfig.BATCH_SIZE
        self.target_update = GRUDQNConfig.TARGET_UPDATE_FREQUENCY
        self.action_size = GRUDQNConfig.ACTION_SIZE
        self.device = torch.device(GRUDQNConfig.DEVICE)

        self.online_net = GRUDQN(input_size, GRUDQNConfig.HIDDEN_SIZE, self.action_size, GRUDQNConfig.NUM_LAYERS).to(
            self.device)
        self.target_net = GRUDQN(input_size, GRUDQNConfig.HIDDEN_SIZE, self.action_size, GRUDQNConfig.NUM_LAYERS).to(
            self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=GRUDQNConfig.LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer()
        self.episode_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            self.online_net.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_t)
                action = q_values.argmax(dim=1).item()
            self.online_net.train()
            return action

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def on_episode_end(self) -> None:
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.online_net.state_dict(), path)