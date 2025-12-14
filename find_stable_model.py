#!/usr/bin/env python3
"""
Поиск самой стабильной модели из базы данных.
Тестирует все сохранённые модели на N игр и выбирает лучшую по стабильности.
"""
import sqlite3
import numpy as np
from env import SnakeEnv
from neural_network import SnakeNetwork

LAYER_SIZES = (36, 20, 12, 4)
GRID_SIZE = 10
TEST_GAMES = 10  # Сколько игр для теста стабильности


def test_model(weights, num_games=TEST_GAMES):
    """Тестирует модель на num_games игр, возвращает статистику"""
    net = SnakeNetwork(LAYER_SIZES)
    net.set_weights_flat(weights)

    scores = []
    wins = 0

    for _ in range(num_games):
        env = SnakeEnv(GRID_SIZE, GRID_SIZE)
        state = env.reset()
        while not env.done:
            action = net.predict(state)
            state, _, _ = env.step(action)

        scores.append(env.get_score())
        if env.is_win():
            wins += 1

    return {
        'wins': wins,
        'win_rate': wins / num_games,
        'min_score': min(scores),
        'max_score': max(scores),
        'avg_score': sum(scores) / len(scores),
        'scores': scores
    }


def find_stable_models(db_path="snake_evolution.db", min_score=90):
    """Ищет стабильные модели в базе данных"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Получаем все модели с высоким score
    cursor.execute('''
        SELECT id, simulation_id, generation, score, steps, weights
        FROM best_snakes
        WHERE score >= ?
        ORDER BY score DESC, steps DESC
    ''', (min_score,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"Не найдено моделей с score >= {min_score}")
        return None

    print(f"Найдено {len(rows)} моделей с score >= {min_score}")
    print(f"Тестируем каждую на {TEST_GAMES} играх...\n")

    results = []

    for i, (id_, sim_id, gen, db_score, steps, weights_blob) in enumerate(rows):
        weights = np.frombuffer(weights_blob, dtype=np.float32)

        # Проверяем размер весов
        expected_size = sum(LAYER_SIZES[i] * LAYER_SIZES[i+1] + LAYER_SIZES[i+1]
                          for i in range(len(LAYER_SIZES)-1))
        if len(weights) != expected_size:
            print(f"  [{i+1}] Пропуск - неверный размер весов ({len(weights)} != {expected_size})")
            continue

        stats = test_model(weights)
        results.append({
            'id': id_,
            'sim_id': sim_id,
            'generation': gen,
            'db_score': db_score,
            **stats,
            'weights': weights
        })

        print(f"  [{i+1}/{len(rows)}] sim={sim_id} gen={gen} | "
              f"Wins: {stats['wins']}/{TEST_GAMES} | "
              f"Scores: {stats['min_score']}-{stats['max_score']} (avg={stats['avg_score']:.1f})")

    if not results:
        print("Нет совместимых моделей!")
        return None

    # Сортируем по количеству побед, потом по минимальному score
    results.sort(key=lambda x: (x['wins'], x['min_score']), reverse=True)

    print("\n" + "="*60)
    print("TOP-5 самых стабильных моделей:")
    print("="*60)

    for i, r in enumerate(results[:5]):
        print(f"{i+1}. sim={r['sim_id']} gen={r['generation']} | "
              f"Wins: {r['wins']}/{TEST_GAMES} ({r['win_rate']*100:.0f}%) | "
              f"Min: {r['min_score']} Avg: {r['avg_score']:.1f}")

    # Сохраняем лучшую
    best = results[0]
    if best['wins'] > 0:
        filename = f"models/stable_wins{best['wins']}_min{best['min_score']}.npy"
        np.save(filename, best['weights'])
        print(f"\nЛучшая модель сохранена: {filename}")
        print(f"Win rate: {best['win_rate']*100:.0f}% ({best['wins']}/{TEST_GAMES})")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    # Ищем модели с score >= 90 (близко к победе)
    results = find_stable_models(min_score=90)

    if results and results[0]['wins'] >= 7:
        print("\n*** ЦЕЛЬ ДОСТИГНУТА: найдена модель с 7+ победами из 10! ***")
