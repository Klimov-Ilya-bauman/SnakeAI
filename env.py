"""
Среда змейки v5 - оптимизации и исправления.
- Исправлена нормализация координат еды
- Оптимизация через set для O(1) проверок
- Относительные координаты + умные награды
"""
import numpy as np
import math
from config import GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT


class SnakeEnv:
    # Направления: вперёд, влево, вправо (относительно головы)
    ACTIONS = [0, 1, 2]  # 0=прямо, 1=влево, 2=вправо

    # Абсолютные направления для расчётов
    DIRECTIONS = [UP, RIGHT, DOWN, LEFT]  # По часовой стрелке

    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.reset()

    def reset(self):
        cx, cy = self.grid_width // 2, self.grid_height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.snake_set = set(self.snake)  # O(1) проверки
        self.dir_idx = 1  # Смотрим вправо (индекс в DIRECTIONS)
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.total_steps = 0
        self.done = False
        return self._get_state()

    def _spawn_food(self):
        empty = [(x, y) for x in range(self.grid_width)
                 for y in range(self.grid_height) if (x, y) not in self.snake_set]
        return empty[np.random.randint(len(empty))] if empty else self.snake[-1]

    def _get_direction(self):
        """Текущее направление"""
        return self.DIRECTIONS[self.dir_idx]

    def _turn_left(self):
        self.dir_idx = (self.dir_idx - 1) % 4

    def _turn_right(self):
        self.dir_idx = (self.dir_idx + 1) % 4

    def _get_state(self):
        """
        5 чисел:
        - 3: расстояние до опасности (прямо, слева, справа) — нормализованное [0, 1]
        - 2: относительные координаты еды — нормализованные [-1, 1]
        """
        head = self.snake[0]
        direction = self._get_direction()

        # Направления: прямо, влево, вправо
        dir_straight = direction
        dir_left = self.DIRECTIONS[(self.dir_idx - 1) % 4]
        dir_right = self.DIRECTIONS[(self.dir_idx + 1) % 4]

        # Расстояние до опасности в каждом направлении
        dist_straight = self._distance_to_danger(head, dir_straight)
        dist_left = self._distance_to_danger(head, dir_left)
        dist_right = self._distance_to_danger(head, dir_right)

        # Нормализуем (макс расстояние = размер сетки)
        max_dist = max(self.grid_width, self.grid_height)
        dist_straight /= max_dist
        dist_left /= max_dist
        dist_right /= max_dist

        # Относительные координаты еды (исправленная нормализация)
        food_rel = self._get_relative_food_pos(head, direction)
        # Координаты могут быть от -grid до +grid, нормализуем в [-1, 1]
        food_x = np.clip(food_rel[0] / self.grid_width, -1.0, 1.0)
        food_y = np.clip(food_rel[1] / self.grid_height, -1.0, 1.0)

        state = np.array([
            dist_straight, dist_left, dist_right,
            food_x, food_y
        ], dtype=np.float32)

        return state

    def _distance_to_danger(self, start, direction):
        """Расстояние до стены или тела в направлении"""
        x, y = start
        dist = 0
        # Тело без хвоста (хвост уйдёт на следующем шаге)
        body_set = self.snake_set - {self.snake[-1]}

        while True:
            x += direction[0]
            y += direction[1]
            dist += 1

            # Стена
            if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                return dist

            # Тело (O(1) проверка)
            if (x, y) in body_set:
                return dist

    def _get_relative_food_pos(self, head, direction):
        """
        Координаты еды относительно головы.
        X = вправо от змейки, Y = вперёд от змейки
        """
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]

        # Поворачиваем координаты в зависимости от направления змейки
        if direction == UP:
            return (dx, -dy)
        elif direction == DOWN:
            return (-dx, dy)
        elif direction == LEFT:
            return (-dy, -dx)
        else:  # RIGHT
            return (dy, dx)

    def _is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if point in self.snake_set:
            return True
        return False

    def step(self, action):
        """
        action: 0=прямо, 1=влево, 2=вправо
        """
        self.steps += 1
        self.total_steps += 1

        # Поворачиваем
        if action == 1:
            self._turn_left()
        elif action == 2:
            self._turn_right()

        # Двигаемся
        direction = self._get_direction()
        head = self.snake[0]
        old_head = head  # Сохраняем для расчёта расстояния
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # Столкновение
        if self._is_collision(new_head):
            self.done = True
            # Ранняя смерть — жёсткое наказание
            if self.total_steps < 15:
                reward = -100.0
            else:
                reward = -10.0
            return self._get_state(), reward, True

        # Добавляем новую голову
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)

        # Еда
        if new_head == self.food:
            self.score += 1
            self.steps = 0  # Сброс счётчика для таймаута

            # Награда растёт с количеством съеденного
            reward = math.sqrt(self.score) * 3.5

            if len(self.snake) >= self.grid_width * self.grid_height:
                self.done = True
                reward += 100.0  # Бонус за победу
            else:
                self.food = self._spawn_food()

            return self._get_state(), reward, self.done

        # Обычный шаг — убираем хвост
        tail = self.snake.pop()
        self.snake_set.remove(tail)

        # Таймаут (слишком долго без еды)
        if self.steps > 100 * len(self.snake):
            self.done = True
            return self._get_state(), -10.0, True

        # Награда за приближение к еде
        head = self.snake[0]
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        if new_dist < old_dist:
            return self._get_state(), 0.1, False  # Ближе - маленький плюс
        else:
            return self._get_state(), -0.1, False  # Дальше - маленький минус

    def get_score(self):
        return self.score

    def is_win(self):
        return len(self.snake) >= self.grid_width * self.grid_height

    @property
    def direction(self):
        """Для совместимости с визуализацией"""
        return self._get_direction()
