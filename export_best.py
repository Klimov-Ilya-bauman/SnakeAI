#!/usr/bin/env python3
"""
Экспорт лучших весов из базы данных в файл.
Запускать если play.py не находит веса.
"""
import os
import numpy as np
from database import SnakeDatabase

def export_best_weights():
    os.makedirs('models', exist_ok=True)

    db = SnakeDatabase()

    # Получаем информацию о лучшей змейке
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT score, generation, simulation_id
        FROM best_snakes
        ORDER BY score DESC, steps DESC
        LIMIT 1
    ''')
    row = cursor.fetchone()

    if row:
        score, gen, sim_id = row
        print(f"Best snake: score={score}, gen={gen}, sim_id={sim_id}")

        # Получаем веса
        weights = db.get_best_weights()
        if weights is not None:
            filename = f"models/best_record_{score}.npy"
            np.save(filename, weights)
            print(f"Saved: {filename} ({len(weights)} params)")
        else:
            print("Weights blob is empty!")
    else:
        print("No snakes found in database!")

    db.close()

if __name__ == "__main__":
    export_best_weights()
