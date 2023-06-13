import pygame
import random
import math
from Constants import *


def SpawnPlant():
    position = (random.randint(0, WINDOW_WIDTH-1),
                random.randint(0, WINDOW_HEIGHT-1))
    size = 7
    growth_rate = .2
    seed_range = 15
    personal_space = 2
    mutation_rate = .5
    max_age = 50
    soil_energy = .2
    solar_energy = .2
    max_energy = 75 * size
    plant_attributes = {
        "start_size": size,
        "growth_rate": growth_rate,
        "seed_range": seed_range,
        "mutation_rate": mutation_rate,
        "personal_space": personal_space,
        "max_age": max_age,
        "soil_energy": soil_energy,
        "solar_energy": solar_energy,
        "max_energy": max_energy
    }
    starting_energy = random.uniform(0, max_energy/2)
    return Plant(position=position, color=pygame.Color(int(255*mutation_rate), int(255*growth_rate), int(seed_range)), attributes=plant_attributes, starting_energy=starting_energy)


class Plant():
    def __init__(self, position, color, attributes, starting_energy=0):
        self.attributes = attributes
        self.position = position
        self.color = color
        self.size = self.attributes['start_size']
        self.age = 0
        self.energy = starting_energy
        self.rect = pygame.draw.circle(
            pygame.display.get_surface(), self.color, self.position, self.size)

    def update(self, plant_objects, soil):
        # Gain energy
        gathered_energy = (soil.absorb_energy(position=self.position,  desired_energy=self.size * self.attributes["soil_energy"]) +
                           self.size * self.attributes["solar_energy"])

        # The plant didn't get enough energy, so bring it closer to death
        if gathered_energy < self.size * self.attributes["growth_rate"]:
            self.age += 1
        if self.age > self.attributes["max_age"]:
            self.die(plant_objects, soil=soil)
            return

        self.energy += gathered_energy
        if self.size < 1:
            self.die(plant_objects, soil)
            return

        # If there is enough energy, either grow or spread seeds
        if self.energy > self.attributes["max_energy"]:
            self.age += 1
            self.energy /= 2
            if random.random() > self.attributes['growth_rate']:
                new_plant = self.spread_seed()
                if new_plant.has_space(plant_objects):
                    plant_objects.append(new_plant)

            else:
                self.size += self.attributes['growth_rate']

    def draw(self, window, draw_position=False):
        self.rect = pygame.draw.circle(
            window, self.color, self.position, self.size)
        if draw_position:
            font = pygame.font.Font(None, 12)
            position_text = font.render(
                f"{self.position[0]}, {self.position[1]}", True, COLOR_RED)
            window.blit(position_text, (self.position[0], self.position[1]))

    def has_space(self, other_plants):
        for plant in other_plants:
            distance = math.sqrt(
                (self.position[0] - plant.position[0])**2 + (self.position[1] - plant.position[1])**2)
            if distance < (self.size + plant.size + self.attributes['personal_space']):
                return False
        return True

    def die(self, plant_objects, soil):
        soil.give_energy(position=self.position,
                         provided_energy=self.energy * self.size + self.size * self.age)
        plant_objects.remove(self)

    def spread_seed(self):
        # Create a new plant near the current plant
        spread_distance = random.uniform(
            1, self.attributes['seed_range']) + self.size*2 + self.attributes['personal_space']
        spread_angle = random.uniform(0, 2 * math.pi)
        spread_x = (
            self.position[0] + int(spread_distance * math.cos(spread_angle))) % WINDOW_WIDTH
        spread_y = (self.position[1] + int(spread_distance *
                    math.sin(spread_angle))) % WINDOW_HEIGHT
        mutated_attributes = self.mutate()
        new_plant = Plant((spread_x, spread_y), pygame.Color(int(255*mutated_attributes["mutation_rate"]), int(
            255*mutated_attributes["growth_rate"]), int(mutated_attributes["seed_range"])), attributes=mutated_attributes)
        return new_plant

    def lose_energy(self, energy):
        lost_energy = min(energy, self.energy)
        self.energy -= lost_energy
        return lost_energy

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

        attributes['growth_rate'] = min(attributes['growth_rate'], 0.9)
        attributes['mutation_rate'] = min(attributes['mutation_rate'], .5)
        attributes['solar_energy'] = min(attributes['solar_energy'], .3)
        attributes['soil_energy'] = min(attributes['soil_energy'], .5)
        attributes['personal_space'] = max(attributes['personal_space'], 0)

        return attributes
