# importing libraries
import pygame
import time
import random
from plant import Plant, SpawnPlant
from creature import Creature, spawn_creature 
from soil import Soil
from Constants import *

random.seed(123)

# Window size
window_x = WINDOW_WIDTH
window_y = WINDOW_HEIGHT

# defining colors
black = COLOR_BLACK
white = COLOR_WHITE
red = COLOR_RED
green = COLOR_GREEN
blue = COLOR_BLUE

# Initialising pygame
pygame.init()
font = pygame.font.Font(None, 24)

# Initialise game window
pygame.display.set_caption('Plant Sim')
game_window = pygame.display.set_mode((window_x, window_y))

# Game clock controller
clock = pygame.time.Clock()

## Create the Soil
soil_object = Soil()

plant_objects = []
## Randomly instantiate plants in the world
for i in range(PLANT_COUNT):
    plant_objects.append(SpawnPlant())

creature_objects = []
## Randomly instantiate creatures in the world
for i in range(CREATURE_COUNT):
    creature_objects.append(spawn_creature())
    
# Game loop
running = True
while running == True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.K_ESCAPE:
            running = False

    if len(creature_objects) > CREATURE_LIMIT:
        creature_objects = creature_objects[-CREATURE_LIMIT:] 
    for creature in creature_objects:
        creature.update(plant_objects, creature_objects)

    # Clear the screen
    game_window.fill((100, 200, 100))
    
    soil_object.draw(game_window)

    # Update the state of the plants
    for plant in plant_objects:
        plant.update(plant_objects)
        plant.draw(game_window)

    max_food_eaten = 0
    for creature in creature_objects:
        if creature.num_food_eaten > max_food_eaten:
            max_food_eaten = creature.num_food_eaten
            creature.color = COLOR_RED
        else:
            creature.color = COLOR_BLACK
        creature.draw(game_window)

    # Render plant and creature counts
    plant_count_text = font.render(f"Plants: {len(plant_objects)}", False, COLOR_RED)
    game_window.blit(plant_count_text, (10, 10))

    creature_count_text = font.render(f"Creatures: {len(creature_objects):,}", False, COLOR_RED)
    game_window.blit(creature_count_text, (10, 24))
    
    creature_count_text = font.render(f"Best Eater: {max_food_eaten}", False, COLOR_RED)
    game_window.blit(creature_count_text, (10, 36))
    
    # Update the display
    pygame.display.flip()
    
# Quit Pygame
pygame.quit()
