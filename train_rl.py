"""
Обучение Snake с помощью Double DQN.

Использование:
    python train_rl.py                    # Новое обучение
    python train_rl.py --continue         # Продолжить с последней модели
    python train_rl.py --model model.pt   # Продолжить с конкретной модели

TensorBoard:
    tensorboard --logdir=logs
"""
import os
import glob
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import SnakeEnv
from dqn_agent import DQNAgent


def find_best_model():
    """Найти лучшую сохранённую модель"""
    models = glob.glob("models/*.pt")
    if not models:
        return None

    # Сортируем по score в имени файла
    best_model = None
    best_score = 0
    for path in models:
        try:
            # Формат: model_score_XX.pt или best_XX.pt
            filename = os.path.basename(path)
            if "best_" in filename:
                score = int(filename.replace("best_", "").replace(".pt", ""))
            elif "score_" in filename:
                score = int(filename.split("score_")[1].split("_")[0])
            else:
                continue
            if score > best_score:
                best_score = score
                best_model = path
        except:
            continue

    return best_model


def train(episodes=50000,
          grid_size=10,
          save_every=500,
          eval_every=100,
          continue_training=False,
          model_path=None):
    """
    Основной цикл обучения DQN.
    """
    # Создаём папки
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # TensorBoard
    log_dir = f"logs/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard: tensorboard --logdir={log_dir}")

    # Среда
    env = SnakeEnv(grid_size, grid_size)
    state_size = 42  # Количество сенсоров
    action_size = 4  # Вверх, вниз, влево, вправо

    # Агент
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=[256, 256, 128],
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )

    # Загрузка модели
    if continue_training:
        if model_path:
            agent.load(model_path)
        else:
            best_model = find_best_model()
            if best_model:
                agent.load(best_model)
            else:
                print("Моделей не найдено, начинаем с нуля")

    # Для победы на 10x10 нужно 98 очков
    win_score = grid_size * grid_size - 2

    print("=" * 60)
    print("DQN Training - Snake AI")
    print("=" * 60)
    print(f"Поле: {grid_size}x{grid_size} (победа = {win_score} очков)")
    print(f"Сеть: {state_size} -> 256 -> 256 -> 128 -> {action_size}")
    print(f"Эпизодов: {episodes}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    print()

    # Статистика
    scores_window = deque(maxlen=100)
    best_score = 0
    best_avg_score = 0
    total_wins = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Выбор действия
            action = agent.select_action(state, training=True)

            # Шаг среды
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Сохраняем опыт
            agent.store_transition(state, action, reward, next_state, done)

            # Обучение
            loss = agent.learn()

            state = next_state

        # Статистика эпизода
        score = env.get_score()
        scores_window.append(score)
        avg_score = np.mean(scores_window)

        if env.is_win():
            total_wins += 1

        # Новый рекорд
        if score > best_score:
            best_score = score
            agent.save(f"models/best_{score}.pt")
            print(f"*** NEW BEST: {score}/{win_score} - Episode {episode}")

        # Лучший средний
        if avg_score > best_avg_score and len(scores_window) >= 100:
            best_avg_score = avg_score

        # TensorBoard
        writer.add_scalar('score/episode', score, episode)
        writer.add_scalar('score/avg_100', avg_score, episode)
        writer.add_scalar('score/best', best_score, episode)
        writer.add_scalar('epsilon', agent.epsilon, episode)
        writer.add_scalar('wins/total', total_wins, episode)
        writer.add_scalar('reward/episode', total_reward, episode)

        # Логи
        if episode % eval_every == 0:
            win_rate = (total_wins / episode) * 100 if episode > 0 else 0
            print(f"Ep {episode:6d} | "
                  f"Score: {score:2d} | "
                  f"Avg: {avg_score:5.1f} | "
                  f"Best: {best_score:2d} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Wins: {total_wins} ({win_rate:.1f}%)")

        # Сохранение
        if episode % save_every == 0:
            agent.save(f"models/checkpoint_{episode}.pt")

    # Финал
    writer.close()
    agent.save("models/final.pt")

    print()
    print("=" * 60)
    print(f"Готово! Лучший результат: {best_score}/{win_score}")
    print(f"Лучший средний: {best_avg_score:.1f}")
    print(f"Всего побед: {total_wins}")
    print("=" * 60)

    return agent


def evaluate(agent, env, num_games=10):
    """Оценка агента без обучения"""
    scores = []
    wins = 0

    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, _, done = env.step(action)

        scores.append(env.get_score())
        if env.is_win():
            wins += 1

    return np.mean(scores), np.min(scores), np.max(scores), wins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake AI with DQN")
    parser.add_argument("--continue", "-c", dest="continue_training",
                        action="store_true", help="Continue from best model")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model to continue from")
    parser.add_argument("--episodes", "-e", type=int, default=50000,
                        help="Number of episodes")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        continue_training=args.continue_training,
        model_path=args.model
    )
