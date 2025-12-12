"""
Обучение змейки генетическим алгоритмом.
По мотивам статьи: https://habr.com/ru/articles/773288/

+ Один информативный TensorBoard график
"""
import os

# ВАЖНО: Отключаем GPU/Metal ДО импорта TensorFlow (для Mac M1/M2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
from datetime import datetime
from genetic import GeneticAlgorithm
from database import SnakeDatabase


def train(epochs=500,
          population_size=2000,
          top_k=20,
          mutation_rate=0.05,
          grid_size=10,
          layer_sizes=(32, 12, 8, 4),
          save_every=10,
          use_tensorboard=True,
          name=None):
    """
    Основной цикл обучения

    epochs: количество поколений
    population_size: размер начальной популяции
    top_k: сколько лучших отбираем
    mutation_rate: вероятность мутации
    grid_size: размер поля (10 = 10x10)
    layer_sizes: архитектура сети
    save_every: сохранять лучших каждые N поколений
    use_tensorboard: включить TensorBoard логирование
    """
    # База данных
    db = SnakeDatabase()

    # TensorBoard
    writer = None
    if use_tensorboard:
        import tensorflow as tf
        log_dir = f"logs/genetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard: tensorboard --logdir={log_dir}")

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

    # Для победы на 10x10 нужно 98 очков (100 клеток - 2 начальных)
    win_score = grid_size * grid_size - 2  # съесть 98 яблок

    print("=" * 60)
    print("Генетический алгоритм - Snake AI")
    print("=" * 60)
    print(f"Симуляция: {name} (id={sim_id})")
    print(f"Поле: {grid_size}x{grid_size} (победа = {win_score} очков)")
    print(f"Популяция: {population_size}")
    print(f"Отбор: TOP-{top_k}")
    print(f"Мутация: {mutation_rate * 100}%")
    print(f"Сеть: {' -> '.join(map(str, layer_sizes))}")
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

    # Папка для моделей
    os.makedirs("models", exist_ok=True)

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

        # TensorBoard - один график со всеми метриками
        if writer:
            import tensorflow as tf
            with writer.as_default():
                tf.summary.scalar('score/best', stats['best_score'], step=stats['generation'])
                tf.summary.scalar('score/avg_top', stats['avg_score'], step=stats['generation'])
                tf.summary.scalar('steps/best', stats['best_steps'], step=stats['generation'])
                tf.summary.scalar('wins/this_gen', stats['wins_this_gen'], step=stats['generation'])
                tf.summary.scalar('wins/total', stats['total_wins'], step=stats['generation'])
                tf.summary.scalar('population', stats['population_size'], step=stats['generation'])
            writer.flush()

        # Новый рекорд - СОХРАНЯЕМ СРАЗУ!
        if stats['best_score'] > best_ever:
            best_ever = stats['best_score']
            pct = (best_ever / win_score) * 100
            print(f"*** NEW BEST: {best_ever}/{win_score} ({pct:.1f}%) - gen {stats['generation']}")

            # Автосохранение лучших весов при новом рекорде
            if ga.best_weights is not None:
                record_path = f"models/best_record_{best_ever}.npy"
                np.save(record_path, ga.best_weights)
                # Также сохраняем лучших змеек в БД
                db.save_best_snakes(sim_id, stats['generation'], top_snakes[:5])

    # Эволюция
    for epoch in range(epochs):
        stats = ga.evolve(callback=on_generation)

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            pct = (stats['best_score'] / win_score) * 100
            wins_info = f"Wins: {stats['total_wins']}" if stats['total_wins'] > 0 else ""
            print(f"Gen {stats['generation']:4d} | "
                  f"Best: {stats['best_score']:2d}/{win_score} ({pct:5.1f}%) | "
                  f"Avg: {stats['avg_score']:5.1f} | "
                  f"Pop: {stats['population_size']:4d} | "
                  f"Time: {elapsed:6.0f}s {wins_info}")

        # Ранняя остановка если достигли победы
        if stats['total_wins'] >= 10:
            print(f"\n*** GOAL REACHED: {stats['total_wins']} wins! ***")
            break

    # Финал
    if writer:
        writer.close()
    db.finish_simulation(sim_id)
    db.close()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Готово! Лучший результат: {best_ever}/{win_score}")
    print(f"Всего побед: {ga.wins}")
    print(f"Время: {elapsed / 60:.1f} минут")
    print(f"Данные сохранены в snake_evolution.db")
    print("=" * 60)

    # Сохраняем лучшие веса
    best_net = ga.get_best_network()
    if best_net:
        os.makedirs("models", exist_ok=True)
        weights_path = f"models/best_{name}.npy"
        np.save(weights_path, ga.best_weights)
        print(f"Веса сохранены: {weights_path}")

    return ga


if __name__ == "__main__":
    # Параметры для победы на 10x10
    # Большая популяция + больше поколений = больше шансов найти хорошее решение
    train(
        epochs=500,           # Больше поколений
        population_size=2000,  # Большая популяция
        top_k=20,             # Больше лучших для разнообразия
        mutation_rate=0.05,   # 5% мутаций
        grid_size=10          # Поле 10x10
    )
