"""
Обучение с визуализацией - по статье Mnih et al. 2013/2015.
Можно смотреть как змейка учится в реальном времени.
"""
import os
import datetime
import numpy as np
import tensorflow as tf
import pygame
from env import SnakeEnv
from agent import DQNAgent
from config import WIDTH, HEIGHT, GRID_SIZE, BACKGROUND, SNAKE, FOOD, GRID, BLACK, WHITE, GREEN


class VisualTrainer:
    """Тренер с визуализацией pygame"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
        pygame.display.set_caption('Snake AI Training - DQN')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18)
        self.big_font = pygame.font.SysFont('arial', 24)

        self.env = SnakeEnv()
        self.agent = DQNAgent(state_size=11, action_size=3)

        # Настройки визуализации
        self.fps = 30
        self.training_speed = 10
        self.paused = False
        self.fast_mode = False

    def draw_grid(self):
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))

    def draw_snake(self):
        for i, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
            color = GREEN if i == 0 else SNAKE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 1)

    def draw_food(self):
        x, y = self.env.food
        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
        pygame.draw.rect(self.screen, FOOD, rect)

    def draw_stats(self, episode, score, avg_score, epsilon, best_score, total_steps):
        panel_rect = pygame.Rect(WIDTH, 0, 200, HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), panel_rect)

        stats = [
            ("Episode:", str(episode)),
            ("Score:", str(score)),
            ("Length:", str(len(self.env.snake))),
            ("", ""),
            ("Avg(100):", f"{avg_score:.2f}"),
            ("Best:", str(best_score)),
            ("", ""),
            ("Epsilon:", f"{epsilon:.3f}"),
            ("Steps:", str(total_steps)),
            ("", ""),
            ("--- Controls ---", ""),
            ("SPACE:", "Pause"),
            ("F:", "Fast mode"),
            ("+/-:", "Speed"),
            ("S:", "Save"),
            ("ESC:", "Quit"),
        ]

        y = 20
        for label, value in stats:
            if label.startswith("---"):
                text = self.font.render(label, True, (150, 150, 150))
                self.screen.blit(text, (WIDTH + 10, y))
            elif label:
                label_text = self.font.render(label, True, WHITE)
                value_text = self.font.render(value, True, GREEN if "Avg" in label else WHITE)
                self.screen.blit(label_text, (WIDTH + 10, y))
                self.screen.blit(value_text, (WIDTH + 100, y))
            y += 22

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self.fast_mode = not self.fast_mode
                    print(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.training_speed = min(100, self.training_speed + 5)
                    print(f"Speed: {self.training_speed}")
                elif event.key == pygame.K_MINUS:
                    self.training_speed = max(1, self.training_speed - 5)
                    print(f"Speed: {self.training_speed}")
                elif event.key == pygame.K_s:
                    self.agent.save("models/manual_save.keras")
                    print("Model saved!")

        return True

    def train(self, episodes=10000):
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join("logs", current_time)
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        print("=" * 60)
        print("DQN Visual Training - Mnih et al. 2013/2015")
        print("=" * 60)
        print(f"tensorboard --logdir=logs")
        print()

        scores = []
        best_score = 0
        episode = 0

        running = True
        while running and episode < episodes:
            episode += 1
            state = self.env.reset()
            done = False

            while not done and running:
                running = self.handle_events()
                if not running:
                    break

                if self.paused:
                    self.screen.fill(BACKGROUND)
                    self.draw_grid()
                    self.draw_snake()
                    self.draw_food()

                    pause_text = self.big_font.render("PAUSED", True, WHITE)
                    self.screen.blit(pause_text, (WIDTH // 2 - 40, HEIGHT // 2))

                    avg = np.mean(scores[-100:]) if scores else 0
                    self.draw_stats(episode, self.env.score, avg,
                                    self.agent.epsilon, best_score, self.agent.total_steps)

                    pygame.display.flip()
                    self.clock.tick(30)
                    continue

                # Шаги обучения
                for _ in range(self.training_speed if self.fast_mode else 1):
                    if done:
                        break

                    action = self.agent.act(state)
                    next_state, reward, done = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    # Обучение на каждом шаге (как в статье)
                    self.agent.replay()

                    # Линейный epsilon decay на каждом шаге (как в статье)
                    self.agent.step_epsilon()

                # Визуализация
                if not self.fast_mode:
                    self.screen.fill(BACKGROUND)
                    self.draw_grid()
                    self.draw_snake()
                    self.draw_food()

                    avg = np.mean(scores[-100:]) if scores else 0
                    self.draw_stats(episode, self.env.score, avg,
                                    self.agent.epsilon, best_score, self.agent.total_steps)

                    pygame.display.flip()
                    self.clock.tick(self.fps)

            # Конец эпизода
            score = self.env.get_score()
            scores.append(score)

            if score > best_score:
                best_score = score
                self.agent.save(f"models/best_{score}.keras")
                print(f"NEW BEST: {score} (ep {episode})")

            avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)

            with summary_writer.as_default():
                tf.summary.scalar('score', score, step=episode)
                tf.summary.scalar('avg_100', avg, step=episode)
                tf.summary.scalar('epsilon', self.agent.epsilon, step=episode)

            if episode % 100 == 0:
                print(f"Ep {episode} | Score: {score} | Avg: {avg:.2f} | "
                      f"Best: {best_score} | ε: {self.agent.epsilon:.3f} | Steps: {self.agent.total_steps}")

        self.agent.save("models/final.keras")
        pygame.quit()

        print()
        print(f"Done! {episode} episodes, Best: {best_score}, Avg: {np.mean(scores[-100:]):.2f}")


if __name__ == "__main__":
    trainer = VisualTrainer()
    trainer.train(episodes=10000)
