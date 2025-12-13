"""
Среда змейки с 32 сенсорами (8 лучей).
По мотивам статьи: https://habr.com/ru/articles/773288/

Матрица мира:
  0 = пусто
  1 = тело змейки
  2 = еда
  4 = стена
  7 = голова
"""
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT


class SnakeEnv:
    # 8 направлений: вверх, вниз, влево, вправо + 4 диагонали
    DIRECTIONS = [
        (0, -1),   # вверх
        (0, 1),    # вниз
        (-1, 0),   # влево
        (1, 0),    # вправо
        (-1, -1),  # влево-вверх
        (1, -1),   # вправо-вверх
        (-1, 1),   # влево-вниз
        (1, 1),    # вправо-вниз
    ]

    # 4 действия: вверх, вниз, влево, вправо
    ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def __init__(self, width=None, height=None):
        self.width = width or GRID_WIDTH
        self.height = height or GRID_HEIGHT
        self.reset()

    def reset(self):
        """Сброс игры"""
        # Матрица мира (без стен внутри - всё поле для змейки)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Стен внутри нет - границы проверяются при движении

        # Змейка в центре (голова + хвост)
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy)]  # голова, хвост
        self.grid[cy, cx] = 7      # голова
        self.grid[cy, cx - 1] = 1  # тело

        # Еда
        self.food = self._spawn_food()
        self.grid[self.food[1], self.food[0]] = 2

        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.done = False
        self.max_steps_without_food = self.width * self.height

        return self._get_state()

    def _spawn_food(self):
        """Случайная позиция для еды (на всём поле 10x10)"""
        empty = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    empty.append((x, y))
        if empty:
            idx = np.random.randint(len(empty))
            return empty[idx]
        return self.snake[-1]  # если нет места

    def _get_state(self):
        """
        32 сенсора:
        - 8 направлений × 3 типа (стена, яблоко, хвост) = 24
        - 4 сектора где яблоко (вверх, вниз, влево, вправо)
        - 4 расстояния до яблока по направлениям
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        state = []

        # 8 лучей: расстояние до границы, яблока, хвоста
        for dx, dy in self.DIRECTIONS:
            dist_wall = 0
            dist_food = 0
            dist_body = 0

            x, y = head_x, head_y
            distance = 0

            while True:
                x += dx
                y += dy
                distance += 1

                # Вышли за границы поля = стена
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    dist_wall = 1.0 / distance
                    break

                cell = self.grid[y, x]

                if cell == 2 and dist_food == 0:  # яблоко
                    dist_food = 1.0 / distance
                elif cell == 1 and dist_body == 0:  # тело
                    dist_body = 1.0 / distance
                    break  # хвост блокирует обзор

            state.extend([dist_wall, dist_food, dist_body])

        # Сектор яблока (бинарные)
        state.append(1.0 if food_y < head_y else 0.0)  # яблоко выше
        state.append(1.0 if food_y > head_y else 0.0)  # яблоко ниже
        state.append(1.0 if food_x < head_x else 0.0)  # яблоко слева
        state.append(1.0 if food_x > head_x else 0.0)  # яблоко справа

        # Расстояние до яблока (нормализованное)
        dx_food = food_x - head_x
        dy_food = food_y - head_y
        max_dist = self.width + self.height

        state.append(max(0, -dy_food) / max_dist)  # расстояние вверх
        state.append(max(0, dy_food) / max_dist)   # расстояние вниз
        state.append(max(0, -dx_food) / max_dist)  # расстояние влево
        state.append(max(0, dx_food) / max_dist)   # расстояние вправо

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        action: 0=вверх, 1=вниз, 2=влево, 3=вправо
        """
        if self.done:
            return self._get_state(), 0, True

        self.steps += 1
        self.steps_without_food += 1

        # Направление
        dx, dy = self.ACTIONS[action]

        # Запрет на движение назад (иначе мгновенная смерть)
        current_dir = self.direction
        if (dx, dy) == (-current_dir[0], -current_dir[1]):
            # Пытается идти назад - продолжаем в текущем направлении
            dx, dy = current_dir

        head_x, head_y = self.snake[0]
        new_x, new_y = head_x + dx, head_y + dy

        # Проверка столкновения
        if (new_x < 0 or new_x >= self.width or
            new_y < 0 or new_y >= self.height):
            self.done = True
            return self._get_state(), -1, True

        cell = self.grid[new_y, new_x]

        if cell == 1:  # тело (стен внутри нет, границы проверены выше)
            self.done = True
            return self._get_state(), -1, True

        # Голодная смерть
        if self.steps_without_food >= self.max_steps_without_food:
            self.done = True
            return self._get_state(), -1, True

        # Обновляем старую голову на тело
        self.grid[head_y, head_x] = 1

        # Двигаем змейку
        self.snake.insert(0, (new_x, new_y))
        self.grid[new_y, new_x] = 7  # новая голова

        # Еда
        reward = 0
        if cell == 2:
            self.score += 1
            self.steps_without_food = 0
            reward = 1

            # Победа? (заполнили всё поле 10x10 = 100 клеток)
            if len(self.snake) >= self.width * self.height:
                self.done = True
                return self._get_state(), 10, True

            # Новая еда
            self.food = self._spawn_food()
            self.grid[self.food[1], self.food[0]] = 2
        else:
            # Убираем хвост
            tail = self.snake.pop()
            self.grid[tail[1], tail[0]] = 0

        return self._get_state(), reward, False

    def get_score(self):
        return self.score

    def is_win(self):
        """Победа = змейка заполнила всё поле (100 клеток для 10x10)"""
        return len(self.snake) >= self.width * self.height

    @property
    def direction(self):
        """Текущее направление (для визуализации)"""
        if len(self.snake) < 2:
            return (1, 0)
        head = self.snake[0]
        neck = self.snake[1]
        return (head[0] - neck[0], head[1] - neck[1])
