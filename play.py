"""
Визуализация игры обученной змейки (DQN).

Использование:
    python play.py                    # Автопоиск лучшей модели
    python play.py models/best_50.pt  # Конкретная модель
"""
import os
import sys
import glob
import pygame
import torch
from env import SnakeEnv
from dqn_agent import DQNAgent
from config import WIDTH, HEIGHT, GRID_SIZE, BACKGROUND, SNAKE, FOOD, GRID, BLACK, WHITE, GREEN


class SnakePlayer:
    def __init__(self, model_path=None):
        pygame.init()

        # Размер поля из конфига
        self.grid_w = WIDTH // GRID_SIZE
        self.grid_h = HEIGHT // GRID_SIZE

        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
        pygame.display.set_caption('Snake AI - DQN')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18)

        self.env = SnakeEnv(self.grid_w, self.grid_h)

        # DQN агент
        self.agent = DQNAgent(
            state_size=42,
            action_size=4,
            hidden_sizes=[256, 256, 128]
        )

        # Загрузка модели
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Loaded: {model_path}")
        else:
            print("No model loaded - using random actions!")

        self.fps = 10
        self.games = 0
        self.total_score = 0
        self.best = 0
        self.wins = 0

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Сетка
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))

        # Граница поля
        pygame.draw.rect(self.screen, (150, 50, 50), (0, 0, WIDTH, HEIGHT), 3)

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

        win_rate = (self.wins / self.games * 100) if self.games > 0 else 0
        avg_score = self.total_score / max(1, self.games)

        stats = [
            f"Score: {self.env.score}",
            f"Length: {len(self.env.snake)}",
            f"Steps: {self.env.steps}",
            "",
            f"Games: {self.games}",
            f"Best: {self.best}",
            f"Avg: {avg_score:.1f}",
            f"Wins: {self.wins} ({win_rate:.1f}%)",
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
                action = self.agent.select_action(state, training=False)
                state, _, done = self.env.step(action)

                if done:
                    self.games += 1
                    score = self.env.score
                    self.total_score += score
                    self.best = max(self.best, score)

                    if self.env.is_win():
                        self.wins += 1
                        print(f"Game {self.games}: WIN!")
                    else:
                        print(f"Game {self.games}: Score {score}")

                    pygame.time.wait(300)
                    state = self.env.reset()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

        if self.games > 0:
            win_rate = (self.wins / self.games) * 100
            print(f"\nResults: {self.games} games")
            print(f"Avg: {self.total_score / self.games:.1f}")
            print(f"Best: {self.best}")
            print(f"Wins: {self.wins} ({win_rate:.1f}%)")


def find_best_model():
    """Найти лучшую модель"""
    os.makedirs("models", exist_ok=True)

    models = glob.glob("models/*.pt")
    if not models:
        return None

    # Ищем best_XX.pt с максимальным score
    best_model = None
    best_score = 0

    for path in models:
        try:
            filename = os.path.basename(path)
            if "best_" in filename:
                score = int(filename.replace("best_", "").replace(".pt", ""))
                if score > best_score:
                    best_score = score
                    best_model = path
        except:
            continue

    # Если не нашли best_, берём самый новый
    if not best_model:
        best_model = max(models, key=os.path.getmtime)

    return best_model


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else find_best_model()

    if not model:
        print("No model found! Train first: python train_rl.py")
        print("Running with random actions...")

    player = SnakePlayer(model)
    player.play(100)
