import pygame
import random
import math
from Constants import *

def SpawnCreature():
    position = (random.randint(0,WINDOW_WIDTH-1), random.randint(0, WINDOW_HEIGHT-1))
    size = random.randint(5,10)
    max_energy = random.randint(150*size,300*size)
    max_speed = random.random()
    mutation_rate = random.uniform(0.1, 0.5)
    max_age = random.randint(10,25)
    plant_efficiency = random.uniform(0.1, 0.5)
    meat_efficiency = random.uniform(0.5, 0.75)
    attributes = {
        "start_size":size,
        "mutation_rate":mutation_rate,
        "max_age":max_age,
        "plant_efficiency":plant_efficiency,
        "meat_efficiency":meat_efficiency,
        "max_energy":max_energy,
        "max_speed":max_speed
    }
    return Creature(position=position, attributes=attributes)

class Creature():
    def __init__(self, position, attributes):
        self.attributes = attributes
        self.position = position
        self.energy = attributes["max_energy"] / 2
        self.size = attributes["start_size"]
        self.facing_dir = random.uniform(0.0, math.pi*2)

    def rotate(self, theta):
        self.facing_dir += theta
        if 0 > self.facing_dir:
            self.facing_dir += 2*math.pi
        elif self.facing_dir > 2*math.pi:
            self.facing_dir -= 2*math.pi
    
    def move(self, speed):
        if speed > self.attributes["max_speed"]:
            speed = self.attributes["max_speed"]
        self.position = ( math.cos(self.facing_dir) * speed + self.position[0],
                         math.sin(self.facing_dir) * speed + self.position[1] )
        
    def update(self, plants):
        rotation = random.uniform(0, .5*math.pi)
        speed = random.uniform(0, self.attributes["max_speed"])
        #self.rotate(rotation)
        self.move(self.attributes["max_speed"])

    def reproduce(self):
        # Create a new plant near the current plant
        spawn_distance = random.uniform(1, 5) + self.size*2
        spawn_angle = random.uniform(0, 2 * math.pi)
        spawn_x = (self.position[0] + int(spawn_distance * math.cos(spawn_angle)))%WINDOW_WIDTH
        spawn_y = (self.position[1] + int(spawn_distance * math.sin(spawn_angle)))%WINDOW_HEIGHT
        mutated_attributes = self.mutate()
        new_creature = Creature((spawn_x, spawn_y), attributes=mutated_attributes)
        return new_creature

    def draw(self, window, draw_position = False):
        rect = pygame.Rect(self.position[0], self.position[1], self.size, self.size)
        pygame.draw.rect(window, COLOR_BLACK, rect)
        if draw_position:
            font = pygame.font.Font(None, 12)   
            position_text = font.render(f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
            window.blit(position_text, (self.position[0], self.position[1]))

    def mutate(self):
        mutate = random.random() < self.attributes['mutation_rate']
        attributes = self.attributes.copy()
        if mutate:
            attr_idx = random.randint(0, len(self.attributes))
            for i, attr in enumerate(self.attributes):
                if i != attr_idx:
                    continue
                else:
                    attributes[attr] = random.gauss(0, attributes['mutation_rate']/10) * attributes[attr] + attributes[attr]

        attributes['mutation_rate'] = min(attributes['mutation_rate'], .2)

        return attributes



