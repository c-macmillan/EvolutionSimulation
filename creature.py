import pygame
import random
import math
import numpy as np
from Constants import *
from brain import Brain


def spawn_creature():
    position = pygame.math.Vector2(random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))

    return Creature(position=position)


class Creature():
    def __init__(self, position, parent_weights=None):
        
        # position variables 
        self.position = position
        self.center = (position[0] + CREATURE_SIZE/2, position[1] + CREATURE_SIZE/2)
        self.facing_vector = pygame.math.Vector2( random.random(), random.random()  ).normalize()

        # energy variables
        self.energy = MAX_ENERGY / 2
        self.size = CREATURE_SIZE 
        self.age = 0
        self.num_food_eaten = 0
        
        # initialize neural network
        if parent_weights:
            self.brain = Brain(parent_weights=parent_weights)
        else:
            self.brain = Brain()

        self.color = COLOR_BLACK

    def rotate(self, theta):
        self.facing_vector = self.facing_vector.rotate(theta)

    def move(self, speed):
        if speed > MAX_SPEED:
            speed = MAX_SPEED 
        self.position = self.facing_vector * speed + self.position
        self.position = pygame.math.Vector2(self.position[0] % WINDOW_WIDTH,
                         self.position[1] % WINDOW_HEIGHT)
        self.center = pygame.math.Vector2(self.position.x + self.size/2, self.position.y + self.size/2)

    def sight(self):
        game_window = pygame.display.get_surface()
        output = []
        food_rects = [food.rect for food in self.food]
        for i in range(NUM_SIGHT_LINES):
            left_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  -i*DEGREE_SPREAD ) * SIGHT_LENGTH)
            right_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  i*DEGREE_SPREAD) * SIGHT_LENGTH)
            seen_food_idx = left_eye_rect.collidelistall(food_rects)
            nearest_distance = self.get_distance_to_food(seen_food_idx, SIGHT_LENGTH)
            if nearest_distance < SIGHT_LENGTH: 
                pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  -i*DEGREE_SPREAD) * SIGHT_LENGTH)
            output.append(  1 - nearest_distance / SIGHT_LENGTH  )

            if i == 0: ## The right eye line and the left eye line are the same
                continue

            else:
                seen_food_idx = right_eye_rect.collidelistall(food_rects)
                nearest_distance = self.get_distance_to_food(seen_food_idx, SIGHT_LENGTH)
                if nearest_distance < SIGHT_LENGTH:
                    pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  i*DEGREE_SPREAD ) * SIGHT_LENGTH )
                output.append(  1 - nearest_distance / SIGHT_LENGTH )
        if len(output) != NUM_SIGHT_LINES * 2 - 1:
            print(f"Sight lines broke there should be {NUM_SIGHT_LINES *2-1} outputs, but only {len(output)} were found")
        return output
            
    def get_distance_to_food(self, seen_food_idx, length):
        nearest_distance = length
        for idx in seen_food_idx:
            food = self.food[idx]
            distance = (food.position - self.center).magnitude()
            if distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance

    def update(self, plants, creatures):
        self.age += 1
        self.energy -= .5

        # die if too old
        brain_input = []
        self.food = plants

        sight_lines = self.sight()
        is_eating = 0
        if self.energy <= 0:
            self.die(creatures)
        elif self.energy >= MAX_ENERGY:
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
                plant.die(plants)
                energy_boost = 100
                self.energy += energy_boost 
                self.color = COLOR_WHITE
                is_eating = True
                self.num_food_eaten += 1
                break
        brain_input.extend(sight_lines)
        brain_input.append(float(is_eating))
        percent_energy = self.energy / MAX_ENERGY 
        brain_input.append(percent_energy)
        rotation, move_speed = self.brain(brain_input)

        self.rotate(rotation)
        self.move(min(MAX_ENERGY, move_speed))

    def reproduce(self):
        spawn_distance = random.uniform(1, 5) + self.size*2
        spawn_angle = random.uniform(0, 2 * math.pi)
        spawn_x = (self.center[0] + int(spawn_distance *
                   math.cos(spawn_angle))) % WINDOW_WIDTH
        spawn_y = (self.center[1] + int(spawn_distance *
                   math.sin(spawn_angle))) % WINDOW_HEIGHT

        # brain
        parent_weights = self.brain.state_dict()

        new_creature = Creature(
            (spawn_x, spawn_y), 
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

    def die(self, creature_objects):
        creature_objects.remove(self)
