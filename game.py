# importing libraries
import pygame
import time
import random
import pickle
from plant import Plant, SpawnPlant
from creature import Creature, spawn_creature 
from Constants import *
import cProfile


random.seed(123)
max_fps = 1000

profiler = cProfile.Profile()
profiler.enable()

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
pygame.display.set_caption('Evolution Simulation')
game_window = pygame.display.set_mode((window_x, window_y))

# Game clock controller
clock = pygame.time.Clock()

plant_objects = []
## Randomly instantiate plants in the world
for i in range(PLANT_COUNT):
    plant_objects.append(SpawnPlant())

creature_objects = []
## Randomly instantiate creatures in the world
for i in range(CREATURE_COUNT):
    creature_objects.append(spawn_creature())

## Save the best creature to see if it actually learned anything
best_creature = None
most_eaten_ever = 0

# Game loop
running = True
while running == True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            best_brain = best_creature.brain
            with open("best_brain.pickle", "wb") as f:
                pickle.dump(best_brain, f)
        elif event.type == pygame.K_ESCAPE:
            running = False
        elif event.type ==  pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                max_fps = 30
                new_gen_creature = Creature(pygame.mouse.get_pos(), parent_weights= best_creature.brain.state_dict() if best_creature else None)
                new_gen_creature.sterile = True
                creature_objects = [spawn_creature()]
            elif event.key == pygame.K_p:
                print("create plant")
                plant_objects.append(SpawnPlant(pygame.mouse.get_pos()))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            new_gen_creature = Creature(pygame.mouse.get_pos(), parent_weights= best_creature.brain.state_dict() if best_creature else None)
            creature_objects.append(new_gen_creature)
            most_eaten_ever = 0

    ## Spawn new creatures and plants if there aren't enough
    if len(creature_objects) < NUM_MIN_CREATURES:
        creature_objects.append(Creature(pygame.Vector2(random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)), parent_weights=best_creature.brain.state_dict() if best_creature else None))
    if len(plant_objects) < NUM_MIN_PLANTS and len(creature_objects) < CREATURE_LIMIT:
        plant_objects.append(SpawnPlant())

    for creature in creature_objects:
        creature.update(plant_objects, creature_objects)

    # Clear the screen
    game_window.fill(COLOR_BROWN)
    
    # Update the state of the plants
    for plant in plant_objects:
        plant.update(plant_objects)
        plant.draw(game_window)

    max_food_eaten = 0
    for creature in creature_objects:
        if creature.num_food_eaten > max_food_eaten:
            max_food_eaten = creature.num_food_eaten
            creature.color = COLOR_RED
            if max_food_eaten > most_eaten_ever:
                print(f"New best creature has eaten: {max_food_eaten}")
                print(creature.brain.state_dict())
                most_eaten_ever = max_food_eaten
                best_creature = creature
        else:
            creature.color = COLOR_BLACK
        creature.draw(game_window)

    # Render plant and creature counts
    plant_count_text = font.render(f"Plants: {len(plant_objects)}", False, COLOR_RED)
    game_window.blit(plant_count_text, (10, 10))

    creature_count_text = font.render(f"Creatures: {len(creature_objects):,}", False, COLOR_RED)
    game_window.blit(creature_count_text, (10, 24))
    
    best_current_eater = font.render(f"Best Eater: {max_food_eaten}", False, COLOR_RED)
    game_window.blit(best_current_eater, (10, 36))

    best_current_eater = font.render(f"Best Eater Ever: {most_eaten_ever}", False, COLOR_RED)
    game_window.blit(best_current_eater, (10, 48))

    best_current_eater = font.render(f"FPS: {clock.get_fps()}", False, COLOR_RED)
    game_window.blit(best_current_eater, (10, 60))
    clock.tick(max_fps)

    
    # Update the display
    pygame.display.flip()
    
# Quit Pygame
profiler.disable()
profiler.print_stats(sort='cumulative')
pygame.quit()
