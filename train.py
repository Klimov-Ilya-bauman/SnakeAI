"""
Обучение змейки генетическим алгоритмом.
По мотивам статьи: https://habr.com/ru/articles/773288/

+ Многопоточность (multiprocessing)
+ Один информативный TensorBoard график
+ Продолжение обучения с лучших весов
"""
import os
import glob

# Отключаем лишние логи TensorFlow (используется только для TensorBoard)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
from datetime import datetime
from genetic import GeneticAlgorithm
from database import SnakeDatabase


def find_best_weights():
    """Найти лучшие сохранённые веса"""
    patterns = ["models/best_record_*.npy"]
    best_score = 0
    best_path = None

    for pattern in patterns:
        for path in glob.glob(pattern):
            # Извлекаем score из имени файла: best_record_25.npy → 25
            try:
                filename = os.path.basename(path)
                score = int(filename.replace("best_record_", "").replace(".npy", ""))
                if score > best_score:
                    best_score = score
                    best_path = path
            except:
                continue

    return best_path, best_score


def train(epochs=500,
          population_size=1000,
          top_k=20,
          mutation_rate=0.15,
          mutation_strength=0.3,
          grid_size=10,
          layer_sizes=(42, 28, 16, 4),
          num_games=5,
          save_every=10,
          use_tensorboard=True,
          continue_training=True,
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
    continue_training: продолжить с лучших весов (если есть)
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
    print(f"Игр на оценку: {num_games} (отбор по минимуму)")
    print(f"Эпох: {epochs}")
    print("=" * 60)
    print()

    # Генетический алгоритм
    ga = GeneticAlgorithm(
        population_size=population_size,
        top_k=top_k,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        layer_sizes=layer_sizes,
        grid_size=grid_size,
        num_games=num_games
    )

    # Загрузка лучших весов (если есть и continue_training=True)
    seed_weights = None
    best_ever = 0

    if continue_training:
        weights_path, prev_score = find_best_weights()
        if weights_path:
            loaded_weights = np.load(weights_path)
            # Проверка совместимости весов с текущей архитектурой
            expected_weights = ga._num_weights
            if len(loaded_weights) == expected_weights:
                seed_weights = loaded_weights
                best_ever = prev_score
                print(f">>> ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ <<<")
                print(f"Загружены веса: {weights_path}")
                print(f"Предыдущий рекорд: {prev_score}/{win_score}")
            else:
                print(f">>> НОВАЯ АРХИТЕКТУРА - ОБУЧЕНИЕ С НУЛЯ <<<")
                print(f"Старые веса ({len(loaded_weights)}) не совместимы с новой сетью ({expected_weights})")
                print(f"Архитектура: {' -> '.join(map(str, layer_sizes))}")
            print()

    # Начальная популяция
    print("Создание начальной популяции...")
    ga.create_initial_population(seed_weights=seed_weights)
    print(f"Создано {len(ga.population)} змеек")
    print(f"Процессов: {ga.num_workers}")
    print()
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
                # Adaptive mutation метрики
                tf.summary.scalar('adaptive/mutation_strength', stats.get('mutation_strength', 0.3), step=stats['generation'])
                tf.summary.scalar('adaptive/plateau_gens', stats.get('plateau_gens', 0), step=stats['generation'])
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
            # Показываем силу мутации если в режиме плато
            mut_info = ""
            if stats.get('plateau_gens', 0) > 10:
                mut_info = f" | Mut: {stats.get('mutation_strength', 0.3):.2f} (plateau: {stats.get('plateau_gens', 0)})"
            print(f"Gen {stats['generation']:4d} | "
                  f"Best: {stats['best_score']:2d}/{win_score} ({pct:5.1f}%) | "
                  f"Avg: {stats['avg_score']:5.1f} | "
                  f"Pop: {stats['population_size']:4d} | "
                  f"Time: {elapsed:6.0f}s {wins_info}{mut_info}")

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
    # Параметры для стабильной победы на 10x10
    # Multi-game evaluation: отбор по минимальному результату из N игр
    # НОВАЯ АРХИТЕКТУРА: 42 входа (6 новых сенсоров для длинной змейки)
    train(
        epochs=10000,          # Больше поколений
        population_size=1000,  # Меньше (каждая играет num_games раз)
        top_k=20,              # Лучших для разнообразия
        mutation_rate=0.15,    # 15% генов мутируют
        mutation_strength=0.3, # Сила мутации (std шума)
        grid_size=10,          # Поле 10x10
        layer_sizes=(42, 28, 16, 4),  # Новая архитектура: +6 сенсоров, больше нейронов
        num_games=5,           # 5 игр на оценку (стабильность!)
        continue_training=True # Продолжаем с лучших весов (если совместимы)
    )
