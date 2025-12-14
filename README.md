# Snake AI - Double DQN

Обучение змейки с помощью Deep Q-Network (Double DQN) на PyTorch.

## Установка

```bash
# Виртуальное окружение
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Зависимости
pip install -r requirements.txt

# Или вручную
pip install torch numpy pygame tensorboard
```

## Файлы

| Файл | Описание |
|------|----------|
| `config.py` | Настройки игры (размер поля, цвета) |
| `env.py` | Среда с 42 сенсорами + reward shaping |
| `dqn_agent.py` | Double DQN агент (PyTorch) |
| `train_rl.py` | Обучение |
| `play.py` | Визуализация обученной модели |

## Обучение

```bash
# Новое обучение
python train_rl.py

# Продолжить с лучшей модели
python train_rl.py --continue

# Продолжить с конкретной модели
python train_rl.py --model models/best_50.pt

# Указать количество эпизодов
python train_rl.py --episodes 100000
```

## Визуализация

```bash
# Автопоиск лучшей модели
python play.py

# Конкретная модель
python play.py models/best_50.pt
```

**Управление:**
- `+/-` — скорость
- `R` — рестарт
- `ESC` — выход

## TensorBoard

```bash
tensorboard --logdir=logs
```
Открой: http://localhost:6006

## Архитектура

### Сенсоры (42 входа)
- 8 лучей × 3 (стена, еда, тело) = 24
- 4 сектора яблока + 4 расстояния = 8
- 4 направление движения = 4
- 2 информация о змейке = 2
- 4 безопасность соседних клеток = 4

### Нейросеть
```
42 → 256 → 256 → 128 → 4
```

### Double DQN
- Target network (обновляется каждые 1000 шагов)
- Experience replay (100k)
- Epsilon-greedy (1.0 → 0.01)
- Huber loss + gradient clipping

## Награды

| Событие | Награда |
|---------|---------|
| Съел еду | +10 |
| Столкновение | -10 |
| Приблизился к еде | +0.1 |
| Отдалился от еды | -0.1 |
| Голодная смерть | -5 |
| Победа (100 клеток) | +100 |

## Цель

Win rate 70%+ на поле 10x10 (победа = 98 яблок).

## Ожидания

- Первые успехи: ~1000 эпизодов
- Стабильная игра: ~10000 эпизодов
- Высокий score: ~50000+ эпизодов
