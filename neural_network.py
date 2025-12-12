"""
Нейросеть для змейки на TensorFlow.
Архитектура: 32 → 12 → 8 → 4 (как в статье)
Функция активации: ReLU
"""
import numpy as np
import tensorflow as tf


class SnakeNetwork:
    """Нейросеть змейки с поддержкой генетики"""

    def __init__(self, layer_sizes=(32, 12, 8, 4)):
        self.layer_sizes = layer_sizes
        self.model = self._build_model()

    def _build_model(self):
        """Создаём модель"""
        model = tf.keras.Sequential()

        # Входной слой
        model.add(tf.keras.layers.Dense(
            self.layer_sizes[1],
            input_dim=self.layer_sizes[0],
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1),
            bias_initializer=tf.keras.initializers.RandomUniform(-1, 1)
        ))

        # Скрытые слои
        for size in self.layer_sizes[2:-1]:
            model.add(tf.keras.layers.Dense(
                size,
                activation='relu',
                kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1),
                bias_initializer=tf.keras.initializers.RandomUniform(-1, 1)
            ))

        # Выходной слой (без активации для softmax потом)
        model.add(tf.keras.layers.Dense(
            self.layer_sizes[-1],
            activation='linear',
            kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1),
            bias_initializer=tf.keras.initializers.RandomUniform(-1, 1)
        ))

        return model

    def predict(self, state):
        """Получить действие (быстрый режим без tf.function)"""
        state = np.reshape(state, (1, -1))
        # Прямой вызов модели быстрее чем predict() для одиночных примеров
        output = self.model(state, training=False).numpy()[0]
        return np.argmax(output)

    def predict_batch(self, states):
        """Предсказание для батча"""
        return self.model.predict(states, verbose=0)

    def get_weights(self):
        """Получить веса как список numpy массивов"""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Установить веса"""
        self.model.set_weights(weights)

    def get_weights_flat(self):
        """Получить веса как плоский массив (для генетики)"""
        weights = self.get_weights()
        return np.concatenate([w.flatten() for w in weights])

    def set_weights_flat(self, flat_weights):
        """Установить веса из плоского массива"""
        weights = self.get_weights()
        idx = 0
        new_weights = []
        for w in weights:
            size = w.size
            new_weights.append(flat_weights[idx:idx + size].reshape(w.shape))
            idx += size
        self.set_weights(new_weights)

    def copy(self):
        """Создать копию сети"""
        new_net = SnakeNetwork(self.layer_sizes)
        new_net.set_weights(self.get_weights())
        return new_net

    def get_total_weights(self):
        """Общее количество весов"""
        return sum(w.size for w in self.get_weights())


def create_random_network(layer_sizes=(32, 12, 8, 4)):
    """Создать сеть со случайными весами"""
    return SnakeNetwork(layer_sizes)
