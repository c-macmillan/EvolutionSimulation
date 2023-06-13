import pygame
import random
import math
from Constants import *


def SpawnCreature():
    position = pygame.math.Vector2(random.randint(0, WINDOW_WIDTH-1),
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
        self.facing_vector = pygame.math.Vector2( random.random(), random.random()  ).normalize()
        self.sight_length = 30

    def rotate(self, theta):
        self.facing_vector = self.facing_vector.rotate(theta)

    def move(self, speed):
        if speed > self.attributes["max_speed"]:
            speed = self.attributes["max_speed"]
        self.position = self.facing_vector * speed + self.position
        self.position = pygame.math.Vector2(self.position[0] % WINDOW_WIDTH,
                         self.position[1] % WINDOW_HEIGHT)
        self.center = pygame.math.Vector2(self.position.x + self.size/2, self.position.y + self.size/2)
        self.energy -= speed * self.size

    def sight(self, num_sight_lines, deg_spread, length):
        game_window = pygame.display.get_surface()
        output = []
        food_rects = [food.rect for food in self.food]
        for i in range(num_sight_lines):
            left_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  -i*deg_spread ) * length)
            right_eye_rect = pygame.draw.line(game_window, COLOR_BLACK, self.center, self.center + self.facing_vector.rotate(  i*deg_spread ) * length)
            seen_food_idx = left_eye_rect.collidelistall(food_rects)
            nearest_distance = self.get_distance_to_food(seen_food_idx, length)
            if nearest_distance < length:
                pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  -i*deg_spread ) * length)
            output.append(  1 - nearest_distance / length  )

            if i == 0: ## The right eye line and the left eye line are the same
                continue

            else:
                seen_food_idx = right_eye_rect.collidelistall(food_rects)
                nearest_distance = self.get_distance_to_food(seen_food_idx, length)
                if nearest_distance < length:
                    pygame.draw.line(game_window, COLOR_WHITE, self.center, self.center + self.facing_vector.rotate(  i*deg_spread ) * length)
                output.append(  1 - nearest_distance / length  )
        print(output)
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
        self.food = plants
        speed = random.uniform(0, self.attributes["max_speed"])
        if self.sight(1, 0, 25) == False:
            rotation = random.uniform(-45, 45)
            self.rotate(rotation)
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
        self.rect = pygame.Rect(
            self.position[0], self.position[1], self.size, self.size)
        self.rect = pygame.draw.rect(window, self.color, self.rect)

        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
            window.blit(position_text, (self.position[0], self.position[1]))

        self.sight(2, 45, 25)

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
