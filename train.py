"""
Обучение змейки генетическим алгоритмом.
По мотивам статьи: https://habr.com/ru/articles/773288/
"""
import os
import time
from datetime import datetime
from genetic import GeneticAlgorithm
from database import SnakeDatabase


def train(epochs=100,
          population_size=1000,
          top_k=15,
          mutation_rate=0.05,
          grid_size=15,
          layer_sizes=(32, 12, 8, 4),
          save_every=10,
          name=None):
    """
    Основной цикл обучения

    epochs: количество поколений
    population_size: размер начальной популяции
    top_k: сколько лучших отбираем
    mutation_rate: вероятность мутации
    grid_size: размер поля
    layer_sizes: архитектура сети
    save_every: сохранять лучших каждые N поколений
    """
    # База данных
    db = SnakeDatabase()

    # Имя симуляции
    if name is None:
        name = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Создаём запись в БД
    sim_id = db.create_simulation(
        name=name,
        grid_size=grid_size,
        population_size=population_size,
        top_k=top_k,
        mutation_rate=mutation_rate,
        layer_sizes=layer_sizes
    )

    print("=" * 60)
    print("Генетический алгоритм - Snake AI")
    print("=" * 60)
    print(f"Симуляция: {name} (id={sim_id})")
    print(f"Поле: {grid_size}x{grid_size}")
    print(f"Популяция: {population_size}")
    print(f"Отбор: TOP-{top_k}")
    print(f"Мутация: {mutation_rate * 100}%")
    print(f"Сеть: {' → '.join(map(str, layer_sizes))}")
    print(f"Эпох: {epochs}")
    print("=" * 60)
    print()

    # Генетический алгоритм
    ga = GeneticAlgorithm(
        population_size=population_size,
        top_k=top_k,
        mutation_rate=mutation_rate,
        layer_sizes=layer_sizes,
        grid_size=grid_size
    )

    # Начальная популяция
    print("Создание начальной популяции...")
    ga.create_initial_population()
    print(f"Создано {len(ga.population)} змеек")
    print()

    best_ever = 0
    start_time = time.time()

    def on_generation(stats, top_snakes):
        nonlocal best_ever

        # Сохраняем в БД
        db.save_generation(
            sim_id,
            stats['generation'],
            stats['best_score'],
            stats['best_steps'],
            stats['avg_score'],
            stats['population_size']
        )

        # Сохраняем лучших периодически
        if stats['generation'] % save_every == 0:
            db.save_best_snakes(sim_id, stats['generation'], top_snakes[:5])

        # Новый рекорд
        if stats['best_score'] > best_ever:
            best_ever = stats['best_score']
            print(f"🏆 NEW BEST: {best_ever} (gen {stats['generation']})")

    # Эволюция
    for epoch in range(epochs):
        stats = ga.evolve(callback=on_generation)

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Gen {stats['generation']:4d} | "
                  f"Best: {stats['best_score']:3d} | "
                  f"Avg: {stats['avg_score']:5.1f} | "
                  f"Pop: {stats['population_size']:4d} | "
                  f"Time: {elapsed:.0f}s")

    # Финал
    db.finish_simulation(sim_id)
    db.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Готово! Лучший результат: {best_ever}")
    print(f"Время: {elapsed / 60:.1f} минут")
    print(f"Данные сохранены в snake_evolution.db")
    print("=" * 60)

    # Сохраняем лучшие веса
    best_net = ga.get_best_network()
    if best_net:
        os.makedirs("models", exist_ok=True)
        weights_path = f"models/best_gen_{name}.npy"
        import numpy as np
        np.save(weights_path, ga.best_weights)
        print(f"Веса сохранены: {weights_path}")

    return ga


if __name__ == "__main__":
    train(
        epochs=100,
        population_size=1000,
        top_k=15,
        mutation_rate=0.05,
        grid_size=15
    )
