"""
Среда змейки - исправленное состояние.
Все направления ОТНОСИТЕЛЬНЫЕ (к голове змейки).
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
        x, y = point
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if point in self.snake_set:
            return True
        return False

    def _get_state(self):
        """
        7 признаков (все ОТНОСИТЕЛЬНЫЕ к направлению змейки):
        - 3: опасность (прямо, слева, справа)
        - 4: еда (впереди, слева, справа, сзади)
        """
        head = self.snake[0]

        # Относительные направления
        dir_forward = self.DIRECTIONS[self.dir_idx]
        dir_left = self.DIRECTIONS[(self.dir_idx - 1) % 4]
        dir_right = self.DIRECTIONS[(self.dir_idx + 1) % 4]
        dir_back = self.DIRECTIONS[(self.dir_idx + 2) % 4]

        # Опасность в 1 клетке
        danger_forward = self._is_danger((head[0] + dir_forward[0], head[1] + dir_forward[1]))
        danger_left = self._is_danger((head[0] + dir_left[0], head[1] + dir_left[1]))
        danger_right = self._is_danger((head[0] + dir_right[0], head[1] + dir_right[1]))

        # Вектор к еде
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]

        # Проверяем направление к еде ОТНОСИТЕЛЬНО змейки
        # Скалярное произведение вектора к еде и направления
        def dot(d):
            return food_dx * d[0] + food_dy * d[1]

        food_forward = dot(dir_forward) > 0  # Еда впереди
        food_left = dot(dir_left) > 0        # Еда слева
        food_right = dot(dir_right) > 0      # Еда справа
        food_back = dot(dir_back) > 0        # Еда сзади

        state = np.array([
            danger_forward, danger_left, danger_right,
            food_forward, food_left, food_right, food_back
        ], dtype=np.float32)

        return state

    def step(self, action):
        """action: 0=прямо, 1=влево, 2=вправо"""
        self.steps += 1

        if action == 1:
            self._turn_left()
        elif action == 2:
            self._turn_right()

        direction = self._get_direction()
        head = self.snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # Смерть
        if self._is_danger(new_head):
            self.done = True
            return self._get_state(), -10.0, True

        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)

        # Еда
        if new_head == self.food:
            self.score += 1
            self.steps = 0

            if len(self.snake) >= self.grid_width * self.grid_height:
                self.done = True
                return self._get_state(), 100.0, True

            self.food = self._spawn_food()
            return self._get_state(), 10.0, False

        # Обычный шаг
        tail = self.snake.pop()
        self.snake_set.remove(tail)

        # Таймаут
        if self.steps > 100 * len(self.snake):
            self.done = True
            return self._get_state(), -10.0, True

        return self._get_state(), 0.0, False

    def get_score(self):
        return self.score

    def is_win(self):
        return len(self.snake) >= self.grid_width * self.grid_height

    @property
    def direction(self):
        return self._get_direction()
