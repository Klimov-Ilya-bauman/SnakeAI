"""
Визуализация игры обученной модели.
"""
import os
import sys
import glob
import pygame
import numpy as np
from env import SnakeEnv
from agent import DQNAgent
from config import WIDTH, HEIGHT, GRID_SIZE, BACKGROUND, SNAKE, FOOD, GRID, BLACK, WHITE, RED, GREEN


class SnakePlayer:
    def __init__(self, model_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
        pygame.display.set_caption('🐍 Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18)

        self.env = SnakeEnv()
        self.agent = DQNAgent(state_size=5, action_size=3)

        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"✅ Loaded: {model_path}")

        self.fps = 10
        self.games = 0
        self.wins = 0
        self.total_score = 0
        self.best = 0

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Grid
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))

        # Snake
        for i, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
            color = (0, 255, 0) if i == 0 else SNAKE
            pygame.draw.rect(self.screen, color, rect)

        # Food
        fx, fy = self.env.food
        rect = pygame.Rect(fx * GRID_SIZE, fy * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
        pygame.draw.rect(self.screen, FOOD, rect)

        # Stats panel
        panel = pygame.Rect(WIDTH, 0, 200, HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), panel)

        stats = [
            f"Score: {self.env.score}",
            f"Length: {len(self.env.snake)}",
            "",
            f"Games: {self.games}",
            f"Best: {self.best}",
            f"Avg: {self.total_score / max(1, self.games):.1f}",
            "",
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
                action = self.agent.act(state, training=False)
                state, _, done = self.env.step(action)

                if done:
                    self.games += 1
                    score = self.env.score
                    self.total_score += score
                    self.best = max(self.best, score)

                    status = "🏆 WIN!" if self.env.is_win() else f"Score: {score}"
                    print(f"Game {self.games}: {status}")

                    pygame.time.wait(500)
                    state = self.env.reset()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

        if self.games > 0:
            print(f"\n📊 Results: {self.games} games, Avg: {self.total_score / self.games:.1f}, Best: {self.best}")


def find_model():
    if not os.path.exists("models"):
        return None

    # Ищем лучшую модель
    for pattern in ["models/final.keras", "models/best_*.keras"]:
        files = glob.glob(pattern)
        if files:
            return sorted(files)[-1]

    return None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else find_model()

    if not model:
        print("❌ No model found! Train first: python train.py")
        sys.exit(1)

    player = SnakePlayer(model)
    player.play(100)