import pygame
import random
import math
from Constants import *


def SpawnPlant(position = None):
    if position:
        position = pygame.math.Vector2(position)
    else: 
        position = pygame.math.Vector2(random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))
    size = PLANT_SIZE
    reproduction_age = PLANT_REPRODUCTION_AGE
    start_age = random.randint(0, reproduction_age)
    seed_range = 15
    personal_space = 2
    plant_attributes = {
        "start_size": size,
        "seed_range": seed_range,
        "personal_space": personal_space,
        "reproduction_age" : reproduction_age
    }
    return Plant(position=position, color=COLOR_GREEN, attributes=plant_attributes, start_age=start_age)


class Plant():
    def __init__(self, position, color, attributes, start_age=0):
        self.attributes = attributes
        self.position = position
        self.color = color
        self.size = self.attributes['start_size']
        self.age = start_age
        self.rect = pygame.draw.circle(
            pygame.display.get_surface(), self.color, self.position, self.size)
        self.bbox = [self.position[0] - self.size, self.position[1] - self.size, 
                     self.position[0] + self.size, self.position[1] + self.size]

    def update(self, plant_objects):
        # If eough time has passed
        self.age += 1
        if self.age > self.attributes['reproduction_age']:
            self.age = 0
            new_plant = self.spread_seed()
            if new_plant.has_space(plant_objects):
                plant_objects.append(new_plant)

    def draw(self, window, draw_position=False):
        self.rect = pygame.draw.circle(
            window, self.color, self.position, self.size)
        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position.x}, {self.position.y}", True, COLOR_RED)
            window.blit(position_text, (self.position.x, self.position.y))

    def has_space(self, other_plants):
        for plant in other_plants:
            distance = math.sqrt(
                (self.position.x - plant.position.x)**2 + (self.position.y - plant.position.y)**2)
            if distance < (self.size + plant.size + self.attributes['personal_space']):
                return False
        return True

    def die(self, plant_objects):
        plant_objects.remove(self)

    def spread_seed(self):
        # Create a new plant near the current plant
        spread_distance = random.uniform(
            1, self.attributes['seed_range']) + self.size*2 + self.attributes['personal_space']
        spread_angle = random.uniform(0, 2 * math.pi)
        spread_x = (
            self.position.x + int(spread_distance * math.cos(spread_angle))) % WINDOW_WIDTH
        spread_y = (self.position.y + int(spread_distance *
                    math.sin(spread_angle))) % WINDOW_HEIGHT
        new_plant = Plant(pygame.math.Vector2(spread_x, spread_y), COLOR_GREEN, attributes=self.attributes)
        return new_plant
