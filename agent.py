"""
DQN Agent - строго по статье Mnih et al. 2013/2015
"Playing Atari with Deep Reinforcement Learning"

Ключевые элементы:
- Experience Replay (random sampling)
- Target Network (обновляется каждые C шагов)
- Линейный epsilon decay (не экспоненциальный)
- Reward clipping [-1, 1]
"""
import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # === Гиперпараметры из статьи ===
        self.gamma = 0.99                    # Discount factor (статья: 0.99)
        self.learning_rate = 0.00025         # Learning rate (статья: 0.00025)
        self.batch_size = 32                 # Minibatch size (статья: 32)
        self.memory_size = 100000            # Replay memory (статья: 1M, уменьшено для змейки)

        # Epsilon: линейный decay от 1.0 до 0.1
        self.epsilon = 1.0                   # Initial exploration (статья: 1.0)
        self.epsilon_min = 0.1               # Final exploration (статья: 0.1)
        self.epsilon_decay_steps = 50000     # Шаги для decay (статья: 1M, уменьшено)

        # Target network update
        self.target_update_freq = 1000       # Каждые C шагов (статья: 10000, уменьшено)

        # Счётчики
        self.train_step = 0
        self.total_steps = 0

        # Replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Сети
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Нейросеть для Q-function"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        # RMSprop использовался в оригинальной статье, но Adam тоже хорошо работает
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=0.95,       # Статья: 0.95
            epsilon=0.01    # Статья: 0.01
        )

        model.compile(optimizer=optimizer, loss='huber')  # Huber = clipped MSE из статьи
        return model

    def update_target_model(self):
        """Копируем веса из online сети в target сеть"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Сохраняем transition в replay memory"""
        # Reward clipping [-1, 1] (как в статье)
        reward = np.clip(reward, -1.0, 1.0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Experience replay - обучение на случайной выборке из памяти"""
        if len(self.memory) < self.batch_size:
            return

        self.train_step += 1

        # Случайная выборка из памяти
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        # Q-values от target network (как в статье)
        target_q = self.target_model.predict(next_states, verbose=0)
        current_q = self.model.predict(states, verbose=0)

        # Bellman update
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                # Q(s,a) = r + γ * max_a' Q_target(s', a')
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(target_q[i])

        # Gradient descent step
        self.model.fit(states, current_q, epochs=1, verbose=0, batch_size=self.batch_size)

        # Update target network каждые C шагов (как в статье)
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()
            print(f"    [Target network updated at step {self.train_step}]")

    def step_epsilon(self):
        """Линейный epsilon decay (как в статье)"""
        self.total_steps += 1
        if self.epsilon > self.epsilon_min:
            # Линейное уменьшение
            decay = (1.0 - self.epsilon_min) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_min, self.epsilon - decay)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()
        self.epsilon = self.epsilon_min
