"""
Double DQN агент для Snake.
Использует PyTorch для максимальной производительности.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQNNetwork(nn.Module):
    """Нейросеть для Q-values"""

    def __init__(self, state_size, action_size, hidden_sizes=[256, 256, 128]):
        super(DQNNetwork, self).__init__()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer с приоритетами (опционально)"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.FloatTensor(np.array([s[0] for s in batch]))
        actions = torch.LongTensor([s[1] for s in batch])
        rewards = torch.FloatTensor([s[2] for s in batch])
        next_states = torch.FloatTensor(np.array([s[3] for s in batch]))
        dones = torch.FloatTensor([s[4] for s in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Double DQN агент с:
    - Target network для стабильности
    - Experience replay
    - Epsilon-greedy exploration
    - Gradient clipping
    """

    def __init__(self,
                 state_size=42,
                 action_size=4,
                 hidden_sizes=[256, 256, 128],
                 lr=0.0005,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9995,
                 buffer_size=100000,
                 batch_size=64,
                 target_update_freq=1000,
                 device=None):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Device (GPU если доступен)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "mps" if torch.backends.mps.is_available()
                                       else "cpu")
        else:
            self.device = device

        print(f"DQN Agent using device: {self.device}")

        # Две сети: policy и target
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Оптимизатор
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Статистика
        self.losses = []

    def select_action(self, state, training=True):
        """Выбор действия (epsilon-greedy)"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Сохранение опыта в буфер"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Один шаг обучения (Double DQN)"""
        if len(self.memory) < self.batch_size:
            return None

        # Сэмплируем batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Текущие Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: используем policy_net для выбора действия,
        # но target_net для оценки Q-value
        with torch.no_grad():
            # Выбираем лучшее действие по policy_net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Оцениваем это действие по target_net
            next_q = self.target_net(next_states).gather(1, next_actions)
            # Bellman equation
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # Loss (Huber loss для стабильности)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Обновляем target network периодически
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path):
        """Сохранение модели"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.learn_step = checkpoint.get('learn_step', 0)
        print(f"Model loaded: {path} (epsilon={self.epsilon:.4f})")
