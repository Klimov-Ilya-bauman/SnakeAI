"""
Среда змейки v6 - проверенный подход.
- 11 бинарных признаков (как в большинстве туториалов)
- Простые награды без shaping
"""
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT


class SnakeEnv:
    ACTIONS = [0, 1, 2]  # 0=прямо, 1=влево, 2=вправо
    DIRECTIONS = [UP, RIGHT, DOWN, LEFT]  # По часовой стрелке

    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.reset()

    def reset(self):
        cx, cy = self.grid_width // 2, self.grid_height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.snake_set = set(self.snake)
        self.dir_idx = 1  # Вправо
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.done = False
        return self._get_state()

    def _spawn_food(self):
        empty = [(x, y) for x in range(self.grid_width)
                 for y in range(self.grid_height) if (x, y) not in self.snake_set]
        return empty[np.random.randint(len(empty))] if empty else self.snake[-1]

    def _get_direction(self):
        return self.DIRECTIONS[self.dir_idx]

    def _turn_left(self):
        self.dir_idx = (self.dir_idx - 1) % 4

    def _turn_right(self):
        self.dir_idx = (self.dir_idx + 1) % 4

    def _is_danger(self, point):
        """Проверка опасности в точке"""
        x, y = point
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if point in self.snake_set:
            return True
        return False

    def _get_state(self):
        """
        11 признаков:
        - 3: опасность в 1 клетке (прямо, слева, справа)
        - 4: текущее направление (one-hot)
        - 4: направление к еде (вверх, вниз, влево, вправо)
        """
        head = self.snake[0]
        direction = self._get_direction()

        # Направления для проверки опасности
        dir_l = self.DIRECTIONS[(self.dir_idx - 1) % 4]
        dir_r = self.DIRECTIONS[(self.dir_idx + 1) % 4]
        dir_s = direction

        # Точки для проверки (1 клетка в каждом направлении)
        point_s = (head[0] + dir_s[0], head[1] + dir_s[1])
        point_l = (head[0] + dir_l[0], head[1] + dir_l[1])
        point_r = (head[0] + dir_r[0], head[1] + dir_r[1])

        # Опасность (бинарные)
        danger_straight = self._is_danger(point_s)
        danger_left = self._is_danger(point_l)
        danger_right = self._is_danger(point_r)

        # Направление движения (one-hot)
        dir_up = direction == UP
        dir_down = direction == DOWN
        dir_left = direction == LEFT
        dir_right = direction == RIGHT

        # Направление к еде
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        state = np.array([
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ], dtype=np.float32)

        return state

    def step(self, action):
        """action: 0=прямо, 1=влево, 2=вправо"""
        self.steps += 1

        # Поворачиваем
        if action == 1:
            self._turn_left()
        elif action == 2:
            self._turn_right()

        # Двигаемся
        direction = self._get_direction()
        head = self.snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # Столкновение = смерть
        if self._is_danger(new_head):
            self.done = True
            return self._get_state(), -10.0, True

        # Добавляем новую голову
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)

        # Еда
        if new_head == self.food:
            self.score += 1
            self.steps = 0

            if len(self.snake) >= self.grid_width * self.grid_height:
                self.done = True
                return self._get_state(), 100.0, True  # Победа

            self.food = self._spawn_food()
            return self._get_state(), 10.0, False

        # Обычный шаг
        tail = self.snake.pop()
        self.snake_set.remove(tail)

        # Таймаут
        if self.steps > 100 * len(self.snake):
            self.done = True
            return self._get_state(), -10.0, True

        return self._get_state(), 0.0, False  # Нейтральный шаг

    def get_score(self):
        return self.score

    def is_win(self):
        return len(self.snake) >= self.grid_width * self.grid_height

    @property
    def direction(self):
        return self._get_direction()
