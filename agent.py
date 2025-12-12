"""
DQN Агент v5 - улучшения по статье DeepMind.
- Huber loss вместо MSE
- Gradient clipping
- Epsilon decay по шагам (не эпизодам)
- Target network update по шагам
- Double DQN
"""
import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size=5, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # Гиперпараметры (ближе к оригинальной статье)
        self.gamma = 0.99  # Было 0.95, в статье 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # В статье 0.1, но для маленькой задачи можно ниже
        self.epsilon_decay = 0.9997  # Баланс между exploration и exploitation
        self.learning_rate = 0.0005
        self.batch_size = 32  # В статье 32

        # Target network update каждые N шагов (в статье ~10000)
        self.target_update_freq = 1000  # Для маленькой задачи меньше
        self.train_step = 0

        self.memory = deque(maxlen=100000)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu',
                                  kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(self.action_size, activation='linear',
                                  kernel_initializer='he_uniform')
        ])

        # Huber loss + gradient clipping (как в статье)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0  # Gradient clipping
        )
        model.compile(optimizer=optimizer, loss='huber')  # Huber вместо MSE
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

        self.train_step += 1

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        # Double DQN: выбираем действие через online сеть, оцениваем через target
        # Оптимизация: один predict для next_states (обе сети)
        next_q_online = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        current_q = self.model.predict(states, verbose=0)

        # Vectorized target calculation
        next_actions = np.argmax(next_q_online, axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * next_q_target[i][next_actions[i]]

        self.model.fit(states, current_q, epochs=1, verbose=0, batch_size=self.batch_size)

        # Epsilon decay на каждом шаге обучения (как в статье)
        self.decay_epsilon()

        # Target network update по шагам (как в статье)
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

    def decay_epsilon(self):
        """Epsilon decay на каждом шаге"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()
        self.epsilon = self.epsilon_min
