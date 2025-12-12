"""
Генетический алгоритм для обучения змейки.
По мотивам статьи: https://habr.com/ru/articles/773288/

Основные операции:
- Селекция (отбор лучших)
- Скрещивание (crossover)
- Мутация
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from env import SnakeEnv
from neural_network import SnakeNetwork


class GeneticAlgorithm:
    def __init__(self,
                 population_size=1000,
                 top_k=15,
                 mutation_rate=0.05,
                 crossover_ratio=0.7,
                 layer_sizes=(32, 12, 8, 4),
                 grid_size=15):
        """
        population_size: размер популяции
        top_k: сколько лучших отбираем
        mutation_rate: вероятность мутации гена
        crossover_ratio: вероятность взять ген от первого родителя (70/30)
        """
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.crossover_ratio = crossover_ratio
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size

        self.population = []
        self.generation = 0
        self.best_score = 0
        self.best_weights = None

    def create_initial_population(self):
        """Создаём начальную популяцию со случайными весами"""
        self.population = []
        for _ in range(self.population_size):
            net = SnakeNetwork(self.layer_sizes)
            self.population.append({
                'weights': net.get_weights_flat(),
                'score': 0,
                'steps': 0
            })
        self.generation = 0

    def evaluate_snake(self, weights):
        """Оценка одной змейки"""
        env = SnakeEnv(self.grid_size, self.grid_size)
        net = SnakeNetwork(self.layer_sizes)
        net.set_weights_flat(weights)

        state = env.reset()
        total_steps = 0

        while not env.done:
            action = net.predict(state)
            state, reward, done = env.step(action)
            total_steps += 1

        return env.get_score(), total_steps

    def evaluate_population(self, parallel=True, max_workers=None):
        """Оценка всей популяции"""
        if parallel:
            workers = max_workers or multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(
                    lambda p: self.evaluate_snake(p['weights']),
                    self.population
                ))
            for i, (score, steps) in enumerate(results):
                self.population[i]['score'] = score
                self.population[i]['steps'] = steps
        else:
            for p in self.population:
                score, steps = self.evaluate_snake(p['weights'])
                p['score'] = score
                p['steps'] = steps

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
        child = np.where(mask, w1, w2)

        return child

    def mutate(self, weights):
        """Мутация"""
        mask = np.random.random(len(weights)) < self.mutation_rate
        mutations = np.random.uniform(-1, 1, len(weights))
        weights = np.where(mask, mutations, weights)
        return weights

    def create_new_generation(self, top_snakes):
        """Создание нового поколения"""
        new_population = []

        # 1. Копии лучших (элита)
        for snake in top_snakes:
            new_population.append({
                'weights': snake['weights'].copy(),
                'score': 0,
                'steps': 0
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
                        'steps': 0
                    })

                    # Потомок с мутацией
                    child_mutated = self.mutate(child.copy())
                    new_population.append({
                        'weights': child_mutated,
                        'score': 0,
                        'steps': 0
                    })

        self.population = new_population
        self.generation += 1

    def evolve(self, callback=None):
        """Один шаг эволюции"""
        # Оценка
        self.evaluate_population()

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
            'population_size': len(self.population)
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
