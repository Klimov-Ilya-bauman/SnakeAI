"""
Нейросеть для змейки на чистом NumPy.
Архитектура: 36 → 20 → 12 → 4 (улучшенная)
Функция активации: ReLU

Без TensorFlow - работает быстрее и поддерживает multiprocessing!
"""
import numpy as np


def relu(x):
    """ReLU активация"""
    return np.maximum(0, x)


class SnakeNetwork:
    """Нейросеть змейки на чистом NumPy"""

    def __init__(self, layer_sizes=(32, 12, 8, 4)):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        """Инициализация случайными весами [-1, 1]"""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.uniform(-1, 1, (self.layer_sizes[i], self.layer_sizes[i + 1])).astype(np.float32)
            b = np.random.uniform(-1, 1, self.layer_sizes[i + 1]).astype(np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def predict(self, state):
        """Получить действие"""
        x = np.array(state, dtype=np.float32)

        # Прямой проход через все слои
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            # ReLU для всех слоёв кроме последнего
            if i < len(self.weights) - 1:
                x = relu(x)

        return np.argmax(x)

    def get_weights_flat(self):
        """Получить веса как плоский массив (для генетики)"""
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten())
            flat.extend(b.flatten())
        return np.array(flat, dtype=np.float32)

    def set_weights_flat(self, flat_weights):
        """Установить веса из плоского массива"""
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            # Веса
            w_size = self.layer_sizes[i] * self.layer_sizes[i + 1]
            self.weights[i] = flat_weights[idx:idx + w_size].reshape(
                self.layer_sizes[i], self.layer_sizes[i + 1]
            ).astype(np.float32)
            idx += w_size

            # Смещения
            b_size = self.layer_sizes[i + 1]
            self.biases[i] = flat_weights[idx:idx + b_size].astype(np.float32)
            idx += b_size

    def get_total_weights(self):
        """Общее количество весов и смещений"""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]  # веса
            total += self.layer_sizes[i + 1]  # смещения
        return total

    def copy(self):
        """Создать копию сети"""
        new_net = SnakeNetwork(self.layer_sizes)
        new_net.set_weights_flat(self.get_weights_flat())
        return new_net
