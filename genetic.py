"""
Генетический алгоритм для обучения змейки.
По мотивам статьи: https://habr.com/ru/articles/773288/

Основные операции:
- Селекция (отбор лучших)
- Скрещивание (crossover)
- Мутация

+ Многопоточность через multiprocessing (NumPy освобождает GIL)
"""
import numpy as np
from multiprocessing import Pool, cpu_count

from env import SnakeEnv
from neural_network import SnakeNetwork


def _evaluate_snake_worker(args):
    """
    Воркер для оценки одной змейки в отдельном процессе.
    Запускает несколько игр для оценки стабильности.
    """
    weights, grid_size, layer_sizes, num_games = args

    net = SnakeNetwork(layer_sizes)
    net.set_weights_flat(weights)

    scores = []
    steps_list = []
    wins = 0

    for _ in range(num_games):
        env = SnakeEnv(grid_size, grid_size)
        state = env.reset()
        while not env.done:
            action = net.predict(state)
            state, _, _ = env.step(action)

        scores.append(env.get_score())
        steps_list.append(env.steps)
        if env.is_win():
            wins += 1

    # Fitness = минимальный score (отбираем стабильных, а не везунчиков)
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)
    total_steps = sum(steps_list)

    # Возвращаем min для отбора, но также avg для статистики
    return min_score, avg_score, total_steps, wins


class GeneticAlgorithm:
    def __init__(self,
                 population_size=2000,
                 top_k=20,
                 mutation_rate=0.15,
                 mutation_strength=0.3,
                 crossover_ratio=0.7,
                 layer_sizes=(36, 20, 12, 4),
                 grid_size=10,
                 num_games=5,
                 num_workers=None):
        """
        population_size: размер популяции
        top_k: сколько лучших отбираем
        mutation_rate: вероятность мутации гена
        mutation_strength: сила мутации (std для нормального распределения)
        crossover_ratio: вероятность взять ген от первого родителя (70/30)
        num_games: количество игр для оценки каждой змейки (для стабильности)
        num_workers: количество процессов (по умолчанию = CPU cores)
        """
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_ratio = crossover_ratio
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size
        self.num_games = num_games
        self.num_workers = num_workers or cpu_count()

        self.population = []
        self.generation = 0
        self.best_score = 0
        self.best_weights = None
        self.wins = 0  # Счётчик побед

        # Adaptive mutation для выхода из плато
        self.base_mutation_strength = mutation_strength
        self.generations_without_improvement = 0
        self.last_best_score = 0

        # Создаём шаблонную сеть для получения структуры весов
        self._template_net = SnakeNetwork(layer_sizes)
        self._num_weights = self._template_net.get_total_weights()

    def create_initial_population(self, seed_weights=None):
        """
        Создаём начальную популяцию.

        seed_weights: если указаны, создаём популяцию на основе этих весов
                      (для продолжения обучения)
        """
        self.population = []

        if seed_weights is not None:
            # Продолжение обучения: создаём вариации лучших весов
            print(f"Загружены веса, создаём вариации...")

            # 1. Оригинал
            self.population.append({
                'weights': seed_weights.copy(),
                'score': 0,
                'steps': 0,
                'win': False
            })

            # 2. Вариации с разной силой мутации
            mutations_rates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
            per_rate = (self.population_size - 1) // len(mutations_rates)

            for rate in mutations_rates:
                for _ in range(per_rate):
                    mutated = seed_weights.copy()
                    mask = np.random.random(len(mutated)) < rate
                    mutations = np.random.uniform(-1, 1, len(mutated)).astype(np.float32)
                    mutated = np.where(mask, mutations, mutated).astype(np.float32)
                    self.population.append({
                        'weights': mutated,
                        'score': 0,
                        'steps': 0,
                        'win': False
                    })

            # 3. Добавляем случайных для разнообразия (10%)
            random_count = self.population_size // 10
            for _ in range(random_count):
                weights = np.random.uniform(-1, 1, self._num_weights).astype(np.float32)
                self.population.append({
                    'weights': weights,
                    'score': 0,
                    'steps': 0,
                    'win': False
                })

            self.best_weights = seed_weights.copy()
        else:
            # Новое обучение: случайные веса
            for _ in range(self.population_size):
                weights = np.random.uniform(-1, 1, self._num_weights).astype(np.float32)
                self.population.append({
                    'weights': weights,
                    'score': 0,
                    'steps': 0,
                    'win': False
                })

        self.generation = 0

    def evaluate_population(self):
        """Оценка всей популяции с многопоточностью (несколько игр на змейку)"""
        # Подготавливаем аргументы для воркеров
        args = [
            (p['weights'], self.grid_size, self.layer_sizes, self.num_games)
            for p in self.population
        ]

        # Параллельная обработка
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_evaluate_snake_worker, args)

        # Обновляем результаты
        wins_this_gen = 0
        for i, (min_score, avg_score, total_steps, wins) in enumerate(results):
            # Используем min_score для отбора (стабильность важнее пиков)
            self.population[i]['score'] = min_score
            self.population[i]['avg_score'] = avg_score
            self.population[i]['steps'] = total_steps
            self.population[i]['wins'] = wins
            self.population[i]['win'] = (wins == self.num_games)  # победа только если все игры выиграны
            wins_this_gen += wins

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
        """
        Soft mutation - добавляем шум вместо полной замены.
        Это сохраняет хорошие решения, слегка их модифицируя.
        """
        mask = np.random.random(len(weights)) < self.mutation_rate
        # Добавляем гауссов шум вместо полной замены
        noise = np.random.normal(0, self.mutation_strength, len(weights)).astype(np.float32)
        weights = weights + mask * noise
        # Ограничиваем веса в разумных пределах
        weights = np.clip(weights, -2.0, 2.0).astype(np.float32)
        return weights

    def create_new_generation(self, top_snakes):
        """Создание нового поколения с сохранением разнообразия и шоком при плато"""
        new_population = []

        # Шок популяции при длительном плато (каждые 50 поколений без улучшения)
        shock_mode = (self.generations_without_improvement > 0 and
                      self.generations_without_improvement % 50 == 0)
        if shock_mode:
            print(f"*** SHOCK! {self.generations_without_improvement} gens without improvement ***")

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

        # 3. Случайные особи для разнообразия
        # При шоке - 30%, обычно - 10%
        random_ratio = 0.30 if shock_mode else 0.10
        random_count = int(self.population_size * random_ratio)
        for _ in range(random_count):
            weights = np.random.uniform(-1, 1, self._num_weights).astype(np.float32)
            new_population.append({
                'weights': weights,
                'score': 0,
                'steps': 0,
                'win': False
            })

        # 4. Дополняем мутациями лучших до нужного размера
        while len(new_population) < self.population_size:
            parent = top_snakes[np.random.randint(len(top_snakes))]
            mutated = self.mutate(parent['weights'].copy())
            new_population.append({
                'weights': mutated,
                'score': 0,
                'steps': 0,
                'win': False
            })

        self.population = new_population
        self.generation += 1

    def evolve(self, callback=None):
        """Один шаг эволюции с adaptive mutation"""
        # Оценка
        wins_this_gen = self.evaluate_population()
        self.wins += wins_this_gen

        # Отбор лучших
        top_snakes = self.select_top()

        # Обновляем лучший результат
        current_best = top_snakes[0]['score']
        if current_best > self.best_score:
            self.best_score = current_best
            self.best_weights = top_snakes[0]['weights'].copy()

        # Adaptive mutation: трекинг плато
        if current_best > self.last_best_score:
            self.last_best_score = current_best
            self.generations_without_improvement = 0
            # Сброс силы мутации при улучшении
            self.mutation_strength = self.base_mutation_strength
        else:
            self.generations_without_improvement += 1

        # Увеличиваем силу мутации при плато
        if self.generations_without_improvement > 10:
            # Плавное увеличение каждые 10 поколений без улучшения
            plateau_factor = 1 + (self.generations_without_improvement - 10) * 0.02
            self.mutation_strength = min(0.8, self.base_mutation_strength * plateau_factor)

        # Статистика
        stats = {
            'generation': self.generation,
            'best_score': current_best,
            'best_steps': top_snakes[0]['steps'],
            'avg_score': np.mean([s['score'] for s in top_snakes]),
            'population_size': len(self.population),
            'wins_this_gen': wins_this_gen,
            'total_wins': self.wins,
            'mutation_strength': self.mutation_strength,
            'plateau_gens': self.generations_without_improvement
        }

        if callback:
            callback(stats, top_snakes)

        # Новое поколение (с возможным шоком)
        self.create_new_generation(top_snakes)

        return stats

    def get_best_network(self):
        """Получить лучшую сеть"""
        if self.best_weights is None:
            return None
        net = SnakeNetwork(self.layer_sizes)
        net.set_weights_flat(self.best_weights)
        return net
