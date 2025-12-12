"""
Генетический алгоритм для обучения змейки.
По мотивам статьи: https://habr.com/ru/articles/773288/

Основные операции:
- Селекция (отбор лучших)
- Скрещивание (crossover)
- Мутация

+ Многопоточность для ускорения (ThreadPool для TensorFlow)
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from env import SnakeEnv
from neural_network import SnakeNetwork


class GeneticAlgorithm:
    def __init__(self,
                 population_size=2000,
                 top_k=20,
                 mutation_rate=0.05,
                 crossover_ratio=0.7,
                 layer_sizes=(32, 12, 8, 4),
                 grid_size=10,
                 num_workers=None):
        """
        population_size: размер популяции
        top_k: сколько лучших отбираем
        mutation_rate: вероятность мутации гена
        crossover_ratio: вероятность взять ген от первого родителя (70/30)
        num_workers: количество потоков (по умолчанию = 8)
        """
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.crossover_ratio = crossover_ratio
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size
        self.num_workers = num_workers or 8  # ThreadPool лучше с умеренным числом

        self.population = []
        self.generation = 0
        self.best_score = 0
        self.best_weights = None
        self.wins = 0  # Счётчик побед

        # Создаём шаблонную сеть для получения структуры весов
        self._template_net = SnakeNetwork(layer_sizes)
        self._num_weights = self._template_net.get_total_weights()

    def create_initial_population(self):
        """Создаём начальную популяцию со случайными весами"""
        self.population = []
        for _ in range(self.population_size):
            # Случайные веса в диапазоне [-1, 1]
            weights = np.random.uniform(-1, 1, self._num_weights).astype(np.float32)
            self.population.append({
                'weights': weights,
                'score': 0,
                'steps': 0,
                'win': False
            })
        self.generation = 0

    def _evaluate_snake(self, weights):
        """Оценка одной змейки"""
        env = SnakeEnv(self.grid_size, self.grid_size)
        net = SnakeNetwork(self.layer_sizes)
        net.set_weights_flat(weights)

        state = env.reset()
        while not env.done:
            action = net.predict(state)
            state, _, _ = env.step(action)

        return env.get_score(), env.steps, env.is_win()

    def evaluate_population(self):
        """Оценка всей популяции с многопоточностью"""
        def evaluate_one(idx):
            weights = self.population[idx]['weights']
            return idx, self._evaluate_snake(weights)

        # Параллельная обработка с ThreadPool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(evaluate_one, range(len(self.population))))

        # Обновляем результаты
        wins_this_gen = 0
        for idx, (score, steps, win) in results:
            self.population[idx]['score'] = score
            self.population[idx]['steps'] = steps
            self.population[idx]['win'] = win
            if win:
                wins_this_gen += 1

        return wins_this_gen

    def select_top(self):
        """Отбор лучших"""
        # Сортируем по очкам (потом по шагам - дольше прожил = лучше)
        self.population.sort(key=lambda x: (x['score'], x['steps']), reverse=True)
        return self.population[:self.top_k]

    def crossover(self, parent1, parent2):
        """Скрещивание двух родителей"""
        w1 = parent1['weights']
        w2 = parent2['weights']

        # Для каждого гена выбираем от какого родителя
        mask = np.random.random(len(w1)) < self.crossover_ratio
        child = np.where(mask, w1, w2).astype(np.float32)

        return child

    def mutate(self, weights):
        """Мутация"""
        mask = np.random.random(len(weights)) < self.mutation_rate
        mutations = np.random.uniform(-1, 1, len(weights)).astype(np.float32)
        weights = np.where(mask, mutations, weights).astype(np.float32)
        return weights

    def create_new_generation(self, top_snakes):
        """Создание нового поколения"""
        new_population = []

        # 1. Копии лучших (элита) - без мутаций
        for snake in top_snakes:
            new_population.append({
                'weights': snake['weights'].copy(),
                'score': 0,
                'steps': 0,
                'win': False
            })

        # 2. Скрещивание каждый с каждым
        for i, p1 in enumerate(top_snakes):
            for j, p2 in enumerate(top_snakes):
                if i != j:
                    # Потомок без мутации
                    child = self.crossover(p1, p2)
                    new_population.append({
                        'weights': child,
                        'score': 0,
                        'steps': 0,
                        'win': False
                    })

                    # Потомок с мутацией
                    child_mutated = self.mutate(child.copy())
                    new_population.append({
                        'weights': child_mutated,
                        'score': 0,
                        'steps': 0,
                        'win': False
                    })

        self.population = new_population
        self.generation += 1

    def evolve(self, callback=None):
        """Один шаг эволюции"""
        # Оценка
        wins_this_gen = self.evaluate_population()
        self.wins += wins_this_gen

        # Отбор лучших
        top_snakes = self.select_top()

        # Обновляем лучший результат
        if top_snakes[0]['score'] > self.best_score:
            self.best_score = top_snakes[0]['score']
            self.best_weights = top_snakes[0]['weights'].copy()

        # Статистика
        stats = {
            'generation': self.generation,
            'best_score': top_snakes[0]['score'],
            'best_steps': top_snakes[0]['steps'],
            'avg_score': np.mean([s['score'] for s in top_snakes]),
            'population_size': len(self.population),
            'wins_this_gen': wins_this_gen,
            'total_wins': self.wins
        }

        if callback:
            callback(stats, top_snakes)

        # Новое поколение
        self.create_new_generation(top_snakes)

        return stats

    def get_best_network(self):
        """Получить лучшую сеть"""
        if self.best_weights is None:
            return None
        net = SnakeNetwork(self.layer_sizes)
        net.set_weights_flat(self.best_weights)
        return net
