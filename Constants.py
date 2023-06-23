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
COLOR_BROWN = pygame.Color(125, 67, 33)

# simulation parameters
CREATURE_COUNT = 20
CREATURE_LIMIT = 500
PLANT_COUNT = 100

# plant attributes
PLANT_REPRODUCTION_AGE = 500


# creature attributes
CREATURE_SIZE = 10
MAX_ENERGY = 100
MAX_SPEED = 50
MUTATION_RATE = .8
PLANT_EFFICIENCY = .01
MEAT_EFFICIENCY = .5

SIGHT_LENGTH = 100
SIZE_MIN_MAX = (5, 10)
DEGREE_SPREAD = 15
NUM_SIGHT_LINES = 3