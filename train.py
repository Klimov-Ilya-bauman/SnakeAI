"""
Обучение змейки v5.
Epsilon decay и target update теперь внутри agent.replay().
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

    print(f"📊 tensorboard --logdir=logs")
    print()

    env = SnakeEnv()
    agent = DQNAgent(state_size=5, action_size=3)

    scores = []
    best = 0
    best_avg = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

        score = env.get_score()
        scores.append(score)
        # epsilon decay и target update теперь внутри replay()

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

        if ep % 50 == 0:
            print(f"Ep {ep:5d} | Score: {score:2d} | Avg: {avg:5.2f} | Best: {best} | ε: {agent.epsilon:.3f}")

        if ep % 500 == 0:
            agent.save(f"models/checkpoint_{ep}.keras")

    agent.save("models/final.keras")
    print(f"\n✅ Готово! Best: {best}, Best Avg: {best_avg:.1f}")


if __name__ == "__main__":
    train(episodes=5000)