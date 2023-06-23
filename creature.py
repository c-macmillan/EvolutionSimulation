import pygame
import random
import math
import numpy as np
from Constants import *
from brain import Brain


def SpawnCreature():
    position = pygame.math.Vector2(random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))
    size = random.randint(*SIZE_MIN_MAX)
    max_energy = random.randint(150*size, 300*size)
    max_speed = random.random() + .1
    mutation_rate = random.uniform(0.1, 0.5)
    plant_efficiency = random.uniform(0.1, 0.5)
    meat_efficiency = random.uniform(0.5, 0.75)

    # temp hardcode
    size = 5 
    max_energy = 1000
    max_speed = .5 
    mutation_rate = .8 
    plant_efficiency = .01
    meat_efficiency = .5 

    attributes = {
        "start_size": size,
        "mutation_rate": mutation_rate,
        "plant_efficiency": plant_efficiency,
        "meat_efficiency": meat_efficiency,
        "max_energy": max_energy,
        "max_speed": max_speed
    }

    return Creature(position=position, attributes=attributes)


class Creature():
    def __init__(self, position, attributes, parent_weights=None):
        self.attributes = attributes
        self.position = position
        self.energy = attributes["max_energy"] / 2
        self.size = attributes["start_size"]
        self.center = (position[0] + self.size/2, position[1] + self.size/2)
        self.facing_vector = pygame.math.Vector2( random.random(), random.random()  ).normalize()
        self.sight_length = SIGHT_LENGTH 
        self.deg_spread = DEGREE_SPREAD
        self.num_sight_lines = NUM_SIGHT_LINES
        self.color = COLOR_BLACK
        self.age = 0
        if parent_weights:
            self.brain = Brain(parent_weights=parent_weights)
        else:
            self.brain = Brain()

    def rotate(self, theta):
        self.facing_vector = self.facing_vector.rotate(theta)
        self.energy -= abs(theta)*.001

    def move(self, speed):
        # print(speed)
        if speed > self.attributes["max_speed"]:
            speed = self.attributes["max_speed"]
        self.position = self.facing_vector * speed + self.position
        self.position = pygame.math.Vector2(self.position[0] % WINDOW_WIDTH,
                         self.position[1] % WINDOW_HEIGHT)
        self.center = pygame.math.Vector2(self.position.x + self.size/2, self.position.y + self.size/2)
        # self.energy -= speed * self.size
        self.energy -= speed 

    def sight(self):
        game_window = pygame.display.get_surface()
        output = []
        food_rects = [food.rect for food in self.food]
        for i in range(self.num_sight_lines):
            left_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  -i*self.deg_spread ) * self.sight_length)
            right_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  i*self.deg_spread ) * self.sight_length)
            seen_food_idx = left_eye_rect.collidelistall(food_rects)
            nearest_distance = self.get_distance_to_food(seen_food_idx, self.sight_length)
            if nearest_distance < self.sight_length:
                pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  -i*self.deg_spread ) * self.sight_length)
            output.append(  1 - nearest_distance / self.sight_length  )

            if i == 0: ## The right eye line and the left eye line are the same
                continue

            else:
                seen_food_idx = right_eye_rect.collidelistall(food_rects)
                nearest_distance = self.get_distance_to_food(seen_food_idx, self.sight_length)
                if nearest_distance < self.sight_length:
                    pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  i*self.deg_spread ) * self.sight_length)
                output.append(  1 - nearest_distance / self.sight_length  )
        if len(output) != self.num_sight_lines * 2 - 1:
            print(f"Sight lines broke there should be {self.num_sight_lines *2-1} outputs, but only {len(output)} were found")
        return output
            
    def get_distance_to_food(self, seen_food_idx, length):
        nearest_distance = length
        for idx in seen_food_idx:
            food = self.food[idx]
            distance = (food.position - self.center).magnitude()
            if distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance

    def update(self, plants, creatures, soil):
        # print(f"Energy: {self.energy / self.attributes['max_energy']*100}")
        self.age += 1
        if self.age > 150:
            creatures.append(self.reproduce())
            if self.energy < self.attributes["max_energy"] / 2:
                self.die(creatures, soil)
            self.energy /= 2
            self.age = 0
        brain_input = []
        self.food = plants
        speed = random.uniform(0, self.attributes["max_speed"])

        sight_lines = self.sight()
        is_eating = 0
        if self.energy < -100:
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
                energy_boost = plant.lose_energy(100)
                energy_boost = 10
                self.energy += energy_boost 
                plant.size -= self.attributes["plant_efficiency"] / (1+speed)
                self.color = COLOR_WHITE
                is_eating = True
                break
        brain_input.extend(sight_lines)
        brain_input.append(float(is_eating))
        percent_energy = self.energy / self.attributes['max_energy']
        brain_input.append(percent_energy)
        rotation, move_speed = self.brain(brain_input)

        # print(f"Rotation: {rotation}")
        self.rotate(rotation)
        self.move(min(self.attributes["max_speed"], move_speed))

    def reproduce(self):
        spawn_distance = random.uniform(1, 5) + self.size*2
        spawn_angle = random.uniform(0, 2 * math.pi)
        spawn_x = (self.center[0] + int(spawn_distance *
                   math.cos(spawn_angle))) % WINDOW_WIDTH
        spawn_y = (self.center[1] + int(spawn_distance *
                   math.sin(spawn_angle))) % WINDOW_HEIGHT

        mutated_attributes = self.mutate()

        # brain
        parent_weights = self.brain.state_dict()

        new_creature = Creature(
            (spawn_x, spawn_y), 
            attributes=mutated_attributes,
            parent_weights=parent_weights)
        return new_creature

    def draw(self, window, draw_position=False):
        self.rect = pygame.Rect(
            self.position[0], self.position[1], self.size, self.size)
        self.rect = pygame.draw.rect(window, self.color, self.rect)

        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
            window.blit(position_text, (self.position[0], self.position[1]))
        # self.sight()

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
