import pygame
import random
import math
from Constants import *


def SpawnCreature():
    position = (random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))
    size = random.randint(5, 10)
    max_energy = random.randint(150*size, 300*size)
    max_speed = random.random() + .1
    mutation_rate = random.uniform(0.1, 0.5)
    max_age = random.randint(10, 25)
    plant_efficiency = random.uniform(0.1, 0.5)
    meat_efficiency = random.uniform(0.5, 0.75)
    attributes = {
        "start_size": size,
        "mutation_rate": mutation_rate,
        "max_age": max_age,
        "plant_efficiency": plant_efficiency,
        "meat_efficiency": meat_efficiency,
        "max_energy": max_energy,
        "max_speed": max_speed
    }

    return Creature(position=position, attributes=attributes)


class Creature():
    def __init__(self, position, attributes):
        self.attributes = attributes
        self.position = position
        self.energy = attributes["max_energy"] / 2
        self.size = attributes["start_size"]
        self.center = (position[0] + self.size/2, position[1] + self.size/2)
        self.facing_dir = random.uniform(0.0, math.pi*2)
        self.sight_length = 30

    def rotate(self, theta):
        self.facing_dir += theta
        if 0 > self.facing_dir:
            self.facing_dir += 2*math.pi
        elif self.facing_dir > 2*math.pi:
            self.facing_dir -= 2*math.pi

    def move(self, speed):
        if speed > self.attributes["max_speed"]:
            speed = self.attributes["max_speed"]
        self.position = (math.cos(self.facing_dir) * speed + self.position[0],
                         math.sin(self.facing_dir) * speed + self.position[1])
        self.position = (self.position[0] % WINDOW_WIDTH,
                         self.position[1] % WINDOW_HEIGHT)
        self.center = (self.position[0] + self.size/2,
                       self.position[1] + self.size/2)
        self.energy -= speed * self.size

    def dir_to_closest_food(self, direction, width, length, visible_objects):
        closest_food = None
        shortest_dist = math.inf
        for obj in visible_objects:
            dist = math.sqrt(
                (self.center[0] - obj.position[0])**2 + (self.center[1] - obj.position[1])**2)
            if dist < shortest_dist:
                closest_food = obj
                shortest_dist = dist
        if shortest_dist < length:
            print(shortest_dist)
            pygame.draw.line(pygame.display.get_surface(),
                             COLOR_BLUE, self.center, obj.position)

        delta_x = self.center[0] - closest_food.position[0]
        delta_y = self.center[1] - closest_food.position[1]
        if delta_y == 0:
            return math.copysign() * math.pi / 2
        if delta_x == 0:
            return math.copysign(delta_y) * math.pi
        return math.atan(delta_x/delta_y)

    def update(self, plants, creatures, soil):
        self.food = plants
        rotation = random.uniform(-.05*math.pi, .05*math.pi)
        speed = random.uniform(0, self.attributes["max_speed"])
        self.rotate(rotation)
        # self.facing_dir = self.dir_to_closest_food(self.facing_dir, 5, 30, plants)
        self.move(self.attributes["max_speed"])
        if self.energy < 0:
            self.die(creatures, soil)
        elif self.energy >= self.attributes["max_energy"]:
            creatures.append(self.reproduce())
            self.energy /= 2
        self.color = COLOR_BLACK
        for plant in plants:
            plant_left_x = plant.position[0]-self.size
            plant_right_x = plant.position[0]+self.size
            plant_top_y = plant.position[1]+self.size
            plant_bottom_y = plant.position[1]-self.size
            # Check four corners of the creature square to see if it's within the plant
            if (plant_left_x < self.center[0] < plant_right_x and plant_top_y > self.center[1] > plant_bottom_y):
                self.energy += plant.lose_energy(
                    self.attributes["plant_efficiency"] * self.size * 5 / (1+speed))
                plant.size -= self.attributes["plant_efficiency"] / (1+speed)
                self.color = COLOR_WHITE
                break

    def reproduce(self):
        spawn_distance = random.uniform(1, 5) + self.size*2
        spawn_angle = random.uniform(0, 2 * math.pi)
        spawn_x = (self.center[0] + int(spawn_distance *
                   math.cos(spawn_angle))) % WINDOW_WIDTH
        spawn_y = (self.center[1] + int(spawn_distance *
                   math.sin(spawn_angle))) % WINDOW_HEIGHT
        mutated_attributes = self.mutate()
        new_creature = Creature(
            (spawn_x, spawn_y), attributes=mutated_attributes)
        return new_creature

    def draw(self, window, draw_position=False):
        rect = pygame.Rect(
            self.position[0], self.position[1], self.size, self.size)
        pygame.draw.rect(window, self.color, rect)

        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
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
                    attributes[attr] = random.gauss(
                        0, attributes['mutation_rate']/10) * attributes[attr] + attributes[attr]

        attributes['mutation_rate'] = min(attributes['mutation_rate'], .2)

        return attributes

    def die(self, creature_objects, soil):
        soil.give_energy(position=self.position,
                         provided_energy=self.energy * self.size + self.size)
        creature_objects.remove(self)
