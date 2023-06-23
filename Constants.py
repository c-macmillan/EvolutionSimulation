import pygame

SPEED = 15
 
# Window size
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 960
 
# defining colors
COLOR_BLACK = pygame.Color(0, 0, 0)
COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_RED = pygame.Color(255, 0, 0)
COLOR_GREEN = pygame.Color(0, 255, 0)
COLOR_BLUE = pygame.Color(0, 0, 255)

# simulation parameters
CREATURE_COUNT = 100
CREATURE_LIMIT = 1000
PLANT_COUNT = 1000


# creature attributes
SIGHT_LENGTH = 100
SIZE_MIN_MAX = (5, 10)
DEGREE_SPREAD = 15
NUM_SIGHT_LINES = 3