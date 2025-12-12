"""
Обучение DQN агента - по статье Mnih et al. 2013/2015.
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from env import SnakeEnv
from agent import DQNAgent


def train(episodes=5000):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_dir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tf.summary.create_file_writer(log_dir)

    print("=" * 60)
    print("DQN Training - по статье Mnih et al. 2013/2015")
    print("=" * 60)
    print(f"📊 tensorboard --logdir=logs")
    print()

    env = SnakeEnv()
    agent = DQNAgent(state_size=11, action_size=3)

    # Вывод гиперпараметров
    print(f"Гиперпараметры:")
    print(f"  γ (gamma):        {agent.gamma}")
    print(f"  Learning rate:    {agent.learning_rate}")
    print(f"  Batch size:       {agent.batch_size}")
    print(f"  Memory size:      {agent.memory_size}")
    print(f"  ε decay steps:    {agent.epsilon_decay_steps}")
    print(f"  Target update:    каждые {agent.target_update_freq} шагов")
    print()

    scores = []
    best = 0
    best_avg = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            # Обучение на каждом шаге (как в статье)
            agent.replay()

            # Линейный epsilon decay на каждом шаге (как в статье)
            agent.step_epsilon()

        score = env.get_score()
        scores.append(score)

        avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)

        if score > best:
            best = score
            agent.save(f"models/best_{score}.keras")
            print(f"🏆 NEW BEST: {score} (ep {ep})")

        if avg > best_avg and len(scores) >= 100:
            best_avg = avg
            agent.save(f"models/best_avg_{avg:.1f}.keras")

        with writer.as_default():
            tf.summary.scalar('score', score, step=ep)
            tf.summary.scalar('avg_100', avg, step=ep)
            tf.summary.scalar('epsilon', agent.epsilon, step=ep)
            tf.summary.scalar('reward', total_reward, step=ep)
            tf.summary.scalar('steps', steps, step=ep)

        if ep % 50 == 0:
            print(f"Ep {ep:5d} | Score: {score:2d} | Avg: {avg:5.2f} | "
                  f"Best: {best} | ε: {agent.epsilon:.3f} | Steps: {agent.total_steps}")

        if ep % 500 == 0:
            agent.save(f"models/checkpoint_{ep}.keras")

    agent.save("models/final.keras")
    print(f"\n✅ Готово! Best: {best}, Best Avg: {best_avg:.1f}")


if __name__ == "__main__":
    train(episodes=5000)
