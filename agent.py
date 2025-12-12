"""
DQN Агент v4 - простая сеть для 5 входов.
"""
import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size=5, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # Гиперпараметры
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Выше минимум - всегда немного исследует
        self.epsilon_decay = 0.999  # Медленнее падает
        self.learning_rate = 0.0005  # Меньше LR - стабильнее
        self.batch_size = 64

        self.memory = deque(maxlen=100000)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        # Double DQN
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        target_q = self.target_model.predict(next_states, verbose=0)
        current_q = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * target_q[i][next_actions[i]]

        self.model.fit(states, current_q, epochs=1, verbose=0, batch_size=self.batch_size)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()
        self.epsilon = self.epsilon_min