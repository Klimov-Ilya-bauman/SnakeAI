"""
Обучение с визуализацией.
Можно смотреть как змейка учится в реальном времени.
"""
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import pygame
from env import SnakeEnv
from agent import DQNAgent
from config import WIDTH, HEIGHT, GRID_SIZE, BACKGROUND, SNAKE, FOOD, GRID, BLACK, WHITE, RED, GREEN


class VisualTrainer:
    """Тренер с визуализацией pygame"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))  # Доп. место для статистики
        pygame.display.set_caption('🐍 Snake AI Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18)
        self.big_font = pygame.font.SysFont('arial', 24)
        
        self.env = SnakeEnv()
        self.agent = DQNAgent()
        
        # Настройки визуализации
        self.fps = 30  # Скорость отрисовки
        self.training_speed = 10  # Шагов обучения между кадрами
        self.paused = False
        self.fast_mode = False  # Быстрый режим без визуализации
        
    def draw_grid(self):
        """Рисуем сетку"""
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID, (0, y), (WIDTH, y))
    
    def draw_snake(self):
        """Рисуем змейку"""
        for i, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
            color = GREEN if i == 0 else SNAKE  # Голова ярче
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 1)
    
    def draw_food(self):
        """Рисуем еду"""
        x, y = self.env.food
        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1)
        pygame.draw.rect(self.screen, FOOD, rect)
    
    def draw_stats(self, episode, score, win_rate, epsilon, best_score, total_wins):
        """Рисуем панель статистики"""
        # Фон панели
        panel_rect = pygame.Rect(WIDTH, 0, 200, HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), panel_rect)
        
        # Статистика
        stats = [
            ("Эпизод:", str(episode)),
            ("Счёт:", str(score)),
            ("Длина:", str(len(self.env.snake))),
            ("", ""),
            ("Win Rate:", f"{win_rate*100:.1f}%"),
            ("Побед:", str(total_wins)),
            ("Best:", str(best_score)),
            ("", ""),
            ("Epsilon:", f"{epsilon:.3f}"),
            ("", ""),
            ("--- Управление ---", ""),
            ("SPACE:", "Пауза"),
            ("F:", "Быстрый режим"),
            ("+/-:", "Скорость"),
            ("S:", "Сохранить"),
            ("ESC:", "Выход"),
        ]
        
        y = 20
        for label, value in stats:
            if label.startswith("---"):
                text = self.font.render(label, True, (150, 150, 150))
                self.screen.blit(text, (WIDTH + 10, y))
            elif label:
                label_text = self.font.render(label, True, WHITE)
                value_text = self.font.render(value, True, GREEN if "Win" in label else WHITE)
                self.screen.blit(label_text, (WIDTH + 10, y))
                self.screen.blit(value_text, (WIDTH + 100, y))
            y += 22
    
    def handle_events(self):
        """Обработка событий"""
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
                    print(f"🚀 Быстрый режим: {'ВКЛ' if self.fast_mode else 'ВЫКЛ'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.training_speed = min(100, self.training_speed + 5)
                    print(f"⚡ Скорость: {self.training_speed}")
                elif event.key == pygame.K_MINUS:
                    self.training_speed = max(1, self.training_speed - 5)
                    print(f"⚡ Скорость: {self.training_speed}")
                elif event.key == pygame.K_s:
                    self.agent.save("models/manual_save.keras")
        
        return True
    
    def train(self, episodes=10000, target_win_rate=0.7):
        """Основной цикл обучения с визуализацией"""
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join("logs", current_time)
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        print(f"📊 TensorBoard: tensorboard --logdir=logs")
        print()
        
        # Статистика
        recent_wins = []
        best_score = 0
        total_wins = 0
        episode = 0
        
        running = True
        while running and episode < episodes:
            episode += 1
            state = self.env.reset()
            done = False
            
            while not done and running:
                # Обработка событий
                running = self.handle_events()
                if not running:
                    break
                
                if self.paused:
                    # Рисуем в паузе
                    self.screen.fill(BACKGROUND)
                    self.draw_grid()
                    self.draw_snake()
                    self.draw_food()
                    
                    pause_text = self.big_font.render("ПАУЗА", True, WHITE)
                    self.screen.blit(pause_text, (WIDTH // 2 - 40, HEIGHT // 2))
                    
                    win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                    self.draw_stats(episode, self.env.score, win_rate, 
                                   self.agent.epsilon, best_score, total_wins)
                    
                    pygame.display.flip()
                    self.clock.tick(30)
                    continue
                
                # Шаг обучения
                for _ in range(self.training_speed if self.fast_mode else 1):
                    if done:
                        break
                    
                    action = self.agent.act(state)
                    next_state, reward, done = self.env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    self.agent.replay()
                
                # Визуализация (если не быстрый режим)
                if not self.fast_mode:
                    self.screen.fill(BACKGROUND)
                    self.draw_grid()
                    self.draw_snake()
                    self.draw_food()
                    
                    win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                    self.draw_stats(episode, self.env.score, win_rate, 
                                   self.agent.epsilon, best_score, total_wins)
                    
                    pygame.display.flip()
                    self.clock.tick(self.fps)
            
            # Конец эпизода
            score = self.env.get_score()
            is_win = self.env.is_win()
            
            if is_win:
                total_wins += 1
                print(f"🏆 ПОБЕДА в эпизоде {episode}! Счёт: {score}")
            
            recent_wins.append(1 if is_win else 0)
            if len(recent_wins) > 100:
                recent_wins.pop(0)
            
            if score > best_score:
                best_score = score
                self.agent.save(f"models/best_{score}.keras")
            
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            
            # TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('score', score, step=episode)
                tf.summary.scalar('win_rate', win_rate, step=episode)
                tf.summary.scalar('epsilon', self.agent.epsilon, step=episode)
            
            # Обновление целевой сети
            if episode % 10 == 0:
                self.agent.update_target_model()
            
            # Прогресс в консоль
            if episode % 100 == 0:
                print(f"Эпизод {episode} | Win rate: {win_rate*100:.1f}% | "
                      f"Epsilon: {self.agent.epsilon:.3f} | Best: {best_score}")
            
            # Проверка цели
            if win_rate >= target_win_rate and len(recent_wins) >= 100:
                print()
                print("🎉 ЦЕЛЬ ДОСТИГНУТА!")
                self.agent.save("models/goal_reached.keras")
                break
        
        # Финал
        self.agent.save("models/final.keras")
        pygame.quit()
        
        print()
        print(f"Итого: {episode} эпизодов, {total_wins} побед, "
              f"Win rate: {win_rate*100:.1f}%, Best: {best_score}")


if __name__ == "__main__":
    # Глобальный агент для доступа из класса
    trainer = VisualTrainer()
    agent = trainer.agent
    trainer.train(episodes=10000)
