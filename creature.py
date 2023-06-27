import pygame
import random
import math
import torch
import numpy as np
from Constants import *
from brain import Brain


def spawn_creature(position = None):
    if position:
        position = pygame.math.Vector2(position)
    else:
        position = pygame.math.Vector2(random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))
    num_layers = 1
    hidden_dim = 5
    masks = [torch.ones(hidden_dim,hidden_dim) for _ in range num_layers]
    return Creature(position=position, masks= masks, num_layers=num_layers, hidden_dim=hidden_dim)


class Creature():
    def __init__(self, position, parent_weights=None, parent_masks=None, num_layers=None, hidden_dim=None, masks=None):
        
        # position variables 
        self.position = position
        self.center = (position[0] + CREATURE_SIZE/2, position[1] + CREATURE_SIZE/2)
        self.facing_vector = pygame.math.Vector2( random.random(), random.random()  ).normalize()

        # energy variables
        self.energy = MAX_ENERGY / 2
        self.size = CREATURE_SIZE 
        self.age = 0
        self.num_food_eaten = 0
        self.sterile = False
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.masks = masks
        # initialize neural network
        if parent_weights:
            self.brain = Brain(input_dim=NUM_SIGHT_LINES*2-1, parent_weights=parent_weights, num_layers=num_layers, hidden_dim=hidden_dim, masks=masks)
        else:
            self.brain = Brain(input_dim=NUM_SIGHT_LINES*2-1, num_layers=num_layers, hidden_dim=hidden_dim, masks=masks)

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
        output = np.zeros(NUM_SIGHT_LINES * 2 - 1, dtype=float)
        food_rects = [food.rect for food in self.food]

        # Pre-calculate rotation angles
        rotation_angles = np.arange(-NUM_SIGHT_LINES + 1, NUM_SIGHT_LINES) * DEGREE_SPREAD

        # Create a matrix of facing vectors rotated by the angles
        rotated_vectors = np.array([self.facing_vector.rotate(angle) for angle in rotation_angles]) * SIGHT_LENGTH

        # Start the drawing and collision checks
        for i in range(len(rotation_angles)):
            rotated_vector = rotated_vectors[i]
            eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + rotated_vector)
            seen_food_idx = eye_rect.collidelistall(food_rects)
            if len(seen_food_idx) > 0:
                pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + rotated_vector)
            output[i] = len(seen_food_idx)

        if len(output) != NUM_SIGHT_LINES * 2 - 1:
            print(f"Sight lines broke there should be {NUM_SIGHT_LINES *2-1} outputs, but only {len(output)} were found")
        return output
            
    def get_distance_to_food(self, seen_food_idx, length):
        nearest_distance = length*2
        for idx in seen_food_idx:
            food = self.food[idx]
            distance = (food.position - self.center).magnitude()
            if distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance
    

    def check_plant_collision(self, plants):
        is_eating = False
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
                self.num_food_eaten += 1
                break

        return is_eating


    def update(self, plants, creatures):
        self.age += 1

        brain_input = []
        self.food = plants

        sight_lines = self.sight()
        if self.energy <= 0:
            self.die(creatures)
        elif self.energy >= MAX_ENERGY and self.num_food_eaten % 3==0 and not (self.sterile):
            creatures.append(self.reproduce())
            self.energy /= 2
        self.color = COLOR_BLACK
        is_eating = self.check_plant_collision(plants)
        brain_input.extend(sight_lines)
        rotation, move_speed = self.brain(brain_input)
        self.energy -= move_speed

        self.rotate(rotation * MAX_TURN_RATE)
        self.move(max(0, move_speed * MAX_SPEED))

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
            parent_weights=parent_weights, num_layers=self.num_layers, hidden_dim=self.hidden_dim, masks=self.masks)
        return new_creature

    def draw(self, window, draw_position=False):
        try:
            self.rect = pygame.Rect(
                self.position.x, self.position.y, self.size, self.size)
            self.rect = pygame.draw.rect(window, self.color, self.rect)
        except Exception as e:
            print(e)
            print(self.position)
            print(type(self.position))
            print(self)

        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
            window.blit(position_text, (self.position[0], self.position[1]))
        self.sight()

    def die(self, creature_objects):
        creature_objects.remove(self)
