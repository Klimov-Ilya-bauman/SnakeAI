"""
SQLite база данных для хранения статистики и весов.
"""
import sqlite3
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class SnakeDatabase:
    def __init__(self, db_path="snake_evolution.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Таблица симуляций
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                grid_size INTEGER,
                population_size INTEGER,
                top_k INTEGER,
                mutation_rate REAL,
                layer_sizes TEXT,
                status TEXT DEFAULT 'running'
            )
        ''')

        # Таблица поколений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                generation INTEGER,
                best_score INTEGER,
                best_steps INTEGER,
                avg_score REAL,
                population_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulation_id) REFERENCES simulations(id)
            )
        ''')

        # Таблица лучших змеек (с весами)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_snakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                generation INTEGER,
                rank INTEGER,
                score INTEGER,
                steps INTEGER,
                weights BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulation_id) REFERENCES simulations(id)
            )
        ''')

        self.conn.commit()

    def create_simulation(self, name, grid_size, population_size, top_k,
                          mutation_rate, layer_sizes):
        """Создать новую симуляцию"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO simulations (name, grid_size, population_size, top_k,
                                    mutation_rate, layer_sizes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, grid_size, population_size, top_k,
              mutation_rate, json.dumps(layer_sizes)))
        self.conn.commit()
        return cursor.lastrowid

    def save_generation(self, simulation_id, generation, best_score, best_steps,
                        avg_score, population_size):
        """Сохранить статистику поколения"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO generations (simulation_id, generation, best_score,
                                    best_steps, avg_score, population_size)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (simulation_id, generation, best_score, best_steps,
              avg_score, population_size))
        self.conn.commit()

    def save_best_snakes(self, simulation_id, generation, top_snakes):
        """Сохранить лучших змеек с весами"""
        cursor = self.conn.cursor()
        for rank, snake in enumerate(top_snakes):
            weights_blob = snake['weights'].tobytes()
            cursor.execute('''
                INSERT INTO best_snakes (simulation_id, generation, rank,
                                        score, steps, weights)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (simulation_id, generation, rank,
                  snake['score'], snake['steps'], weights_blob))
        self.conn.commit()

    def get_best_weights(self, simulation_id=None):
        """Получить веса лучшей змейки"""
        cursor = self.conn.cursor()
        if simulation_id:
            cursor.execute('''
                SELECT weights FROM best_snakes
                WHERE simulation_id = ?
                ORDER BY score DESC, steps DESC
                LIMIT 1
            ''', (simulation_id,))
        else:
            cursor.execute('''
                SELECT weights FROM best_snakes
                ORDER BY score DESC, steps DESC
                LIMIT 1
            ''')
        row = cursor.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def get_simulation_stats(self, simulation_id):
        """Получить статистику симуляции"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT generation, best_score, avg_score
            FROM generations
            WHERE simulation_id = ?
            ORDER BY generation
        ''', (simulation_id,))
        return cursor.fetchall()

    def get_all_simulations(self):
        """Получить все симуляции"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, name, created_at, grid_size, status,
                   (SELECT MAX(best_score) FROM generations WHERE simulation_id = simulations.id) as max_score,
                   (SELECT MAX(generation) FROM generations WHERE simulation_id = simulations.id) as max_gen
            FROM simulations
            ORDER BY created_at DESC
        ''')
        return cursor.fetchall()

    def finish_simulation(self, simulation_id):
        """Отметить симуляцию как завершённую"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE simulations SET status = 'finished' WHERE id = ?
        ''', (simulation_id,))
        self.conn.commit()

    def close(self):
        """Закрыть соединение"""
        if self.conn:
            self.conn.close()
