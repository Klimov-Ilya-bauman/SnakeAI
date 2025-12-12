"""
Визуализация игры обученной змейки.
Генетический алгоритм.
"""
import os
import sys
import glob
import pygame
import numpy as np
from env import SnakeEnv
from neural_network import SnakeNetwork
from database import SnakeDatabase
from config import WIDTH, HEIGHT, GRID_SIZE, BACKGROUND, SNAKE, FOOD, GRID, BLACK, WHITE, GREEN


class SnakePlayer:
    def __init__(self, weights_path=None):
        pygame.init()

        # Размер поля из конфига
        self.grid_w = WIDTH // GRID_SIZE
        self.grid_h = HEIGHT // GRID_SIZE

        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
        pygame.display.set_caption('Snake AI - Genetic')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18)

        self.env = SnakeEnv(self.grid_w, self.grid_h)
        self.net = SnakeNetwork((32, 12, 8, 4))

        # Загрузка весов
        if weights_path and os.path.exists(weights_path):
            weights = np.load(weights_path)
            self.net.set_weights_flat(weights)
            print(f"Loaded: {weights_path}")
        else:
            print("Using random weights")

        self.fps = 10
        self.games = 0
        self.total_score = 0
        self.best = 0

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Сетка
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))

        # Стены (по периметру)
        wall_color = (100, 100, 100)
        for x in range(self.grid_w):
            pygame.draw.rect(self.screen, wall_color,
                           (x * GRID_SIZE, 0, GRID_SIZE - 1, GRID_SIZE - 1))
            pygame.draw.rect(self.screen, wall_color,
                           (x * GRID_SIZE, (self.grid_h - 1) * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))
        for y in range(self.grid_h):
            pygame.draw.rect(self.screen, wall_color,
                           (0, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))
            pygame.draw.rect(self.screen, wall_color,
                           ((self.grid_w - 1) * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))

        # Змейка
        for i, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
            color = GREEN if i == 0 else SNAKE
            pygame.draw.rect(self.screen, color, rect)

        # Еда
        fx, fy = self.env.food
        rect = pygame.Rect(fx * GRID_SIZE, fy * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
        pygame.draw.rect(self.screen, FOOD, rect)

        # Панель статистики
        panel = pygame.Rect(WIDTH, 0, 200, HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), panel)

        stats = [
            f"Score: {self.env.score}",
            f"Length: {len(self.env.snake)}",
            f"Steps: {self.env.steps}",
            "",
            f"Games: {self.games}",
            f"Best: {self.best}",
            f"Avg: {self.total_score / max(1, self.games):.1f}",
            "",
            "Controls:",
            "+/- Speed",
            "R Restart",
            "ESC Quit"
        ]

        for i, text in enumerate(stats):
            surf = self.font.render(text, True, WHITE)
            self.screen.blit(surf, (WIDTH + 10, 20 + i * 25))

        pygame.display.flip()

    def play(self, num_games=100):
        state = self.env.reset()
        running = True

        while running and self.games < num_games:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        state = self.env.reset()
                    elif event.key == pygame.K_EQUALS:
                        self.fps = min(60, self.fps + 2)
                    elif event.key == pygame.K_MINUS:
                        self.fps = max(2, self.fps - 2)

            if not self.env.done:
                action = self.net.predict(state)
                state, _, done = self.env.step(action)

                if done:
                    self.games += 1
                    score = self.env.score
                    self.total_score += score
                    self.best = max(self.best, score)

                    status = "WIN!" if self.env.is_win() else f"Score: {score}"
                    print(f"Game {self.games}: {status}")

                    pygame.time.wait(500)
                    state = self.env.reset()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

        if self.games > 0:
            print(f"\nResults: {self.games} games, Avg: {self.total_score / self.games:.1f}, Best: {self.best}")


def find_weights():
    """Найти файл с весами"""
    # Из БД
    try:
        db = SnakeDatabase()
        weights = db.get_best_weights()
        db.close()
        if weights is not None:
            # Сохраняем временно
            np.save("models/temp_best.npy", weights)
            return "models/temp_best.npy"
    except:
        pass

    # Из файлов
    patterns = ["models/best_gen_*.npy", "models/*.npy"]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return sorted(files)[-1]

    return None


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else find_weights()

    if not weights:
        print("No weights found! Train first: python train.py")
        print("Or run with random weights anyway...")

    player = SnakePlayer(weights)
    player.play(100)
