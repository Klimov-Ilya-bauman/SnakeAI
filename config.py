# Настройки игры
WIDTH = 450   # 15 клеток * 30
HEIGHT = 450

# Сетка
GRID_SIZE = 30
GRID_WIDTH = WIDTH // GRID_SIZE   # 15 клеток
GRID_HEIGHT = HEIGHT // GRID_SIZE  # 15 клеток

# Цвета
BLUE = (0, 139, 139)
GREEN = (124, 252, 0)
RED = (255, 0, 0)  # Сделал ярче
GRAY = (102, 205, 170)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SCORE_BG_COLOR = (255, 255, 255, 180)

SNAKE = GREEN
FONT = BLACK
FOOD = RED
GRID = GRAY
BACKGROUND = BLUE
TEXT_COLOR = BLACK

# Направления
UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

# Скорость
FPS = 5  # Увеличил для более плавной игры

# Начальная длина змейки
INITIAL_SNAKE_LENGTH = 3  # Увеличил для лучшего старта

# Очки за еду
SCORE_FOR_FOOD = 10