from Constants import *
import random
import pygame

cell_size = 60


class Soil():
    def __init__(self):
        self.num_rows = WINDOW_HEIGHT // cell_size
        self.num_cols = WINDOW_WIDTH // cell_size
        self._energy = {(c, r):  random.randint(100, 300)
                        for r in range(self.num_rows) for c in range(self.num_cols)}
        self.max_energy = 500

    def absorb_energy(self, position, desired_energy):
        available_energy = self._energy[(
            position[0]//cell_size, position[1]//cell_size)]
        absorbed_energy = min(available_energy, desired_energy)
        self._energy[(position[0]//cell_size, position[1]//cell_size)
                     ] = available_energy - absorbed_energy
        return absorbed_energy

    def give_energy(self, position, provided_energy):
        self._energy[(position[0]//cell_size, position[1] //
                      cell_size)] += provided_energy

    def draw(self, screen, draw_energy=False):
        total_energy = 0
        for pos, energy in self._energy.items():
            font = pygame.font.Font(None, 12)
            rect = pygame.Rect(pos[0] * cell_size, pos[1]
                               * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, pygame.Color((139, 69, 19, 0)), rect)
