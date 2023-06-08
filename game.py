# importing libraries
import pygame
import time
import random
from plant import Plant, SpawnPlant
from creature import Creature, SpawnCreature
from soil import Soil
from Constants import *

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
## Randomly instantiate 1000 plants in the world
for i in range(1000):
    plant_objects.append(SpawnPlant())

creature_objects = []
## Randomly instantiate 1000 plants in the world
for i in range(10):
    creature_objects.append(SpawnCreature())
    
# Game loop
running = True
while running == True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.K_ESCAPE:
            running = False
    
    # Update the state of the plants
    for obj in plant_objects:
        obj.update(plant_objects, soil_object)
    for obj in creature_objects:
        obj.update(plant_objects)
    
    # Clear the screen
    game_window.fill((100, 200, 100))
    
    soil_object.draw(game_window)
    # Draw the plants on the screen
    for plant in plant_objects:
        plant.draw(game_window)
    for creature in creature_objects:
        creature.draw(game_window)

    # Render plant and creature counts
    plant_count_text = font.render(f"Plants: {len(plant_objects)}", True, COLOR_RED)
    game_window.blit(plant_count_text, (10, 10))
    plant_energy = 0
    for plant in plant_objects:
        plant_energy += plant.energy

    plant_count_text = font.render(f"Plant Energy: {round(plant_energy):,}", True, COLOR_RED)
    game_window.blit(plant_count_text, (10, 24))
    
    
    # Update the display
    pygame.display.flip()
    
    # Control the frame rate
    # clock.tick(1000)

# Quit Pygame
pygame.quit()
