# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2 Problem Domain

This code provides a basic and interactive grid world environment where a robot can navigate using the arrow keys. The robot encounters walls that block movement, gold that gives positive rewards, and traps that give negative rewards. The game ends when the robot reaches its goal. The robot's score reflects the rewards it collects and penalties it incurs.

"""

import pygame
import numpy as np
import random
# Constants for our display
GRID_SIZE = 10  # Easily change this value
CELL_SIZE = 60  # Adjust this based on your display preferences
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
GOLD_REWARD = 10 
TRAP_PENALTY = -10 
ROBOT_COLOR = (0, 128, 255)
GOAL_COLOR = (0, 255, 0)
WALL_COLOR = (0, 0, 0)
EMPTY_COLOR = (255, 255, 255)
GOLD_COLOR = (255, 255, 0)  # Yellow
TRAP_COLOR = (255, 0, 0)   # Red

random.seed(100)

class GridWorld:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size))
        # Randomly select start and goal positions
        self.start = (random.randint(0, size-1), random.randint(0, size-1))
        self.goal = (random.randint(0, size-1), random.randint(0, size-1))
        self.robot_pos = self.start
        self.score = 0
        self.generate_walls_traps_gold()

    def generate_walls_traps_gold(self):
        
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.start and (i, j) != self.goal:
                    rand_num = random.random()
                    if rand_num < 0.1:  # 10% chance for a wall
                        self.grid[i][j] = np.inf
                    elif rand_num < 0.2:  # 20% chance for gold
                        self.grid[i][j] = GOLD_REWARD
                    elif rand_num < 0.3:  # 30% chance for a trap
                        self.grid[i][j] = TRAP_PENALTY

    def move(self, direction):
        """Move the robot in a given direction."""
        x, y = self.robot_pos
        # Conditions check for boundaries and walls
        if direction == "up" and x > 0 and self.grid[x-1][y] != np.inf:
            x -= 1
        elif direction == "down" and x < self.size-1 and self.grid[x+1][y] != np.inf:
            x += 1
        elif direction == "left" and y > 0 and self.grid[x][y-1] != np.inf:
            y -= 1
        elif direction == "right" and y < self.size-1 and self.grid[x][y+1] != np.inf:
            y += 1
        reward = self.grid[x][y] - 1  # step penalty
        self.robot_pos = (x, y)
        self.grid[x][y] = 0  # Clear the cell after the robot moves
        self.score += reward
        return reward

    def display(self):
        """Print a text-based representation of the grid world (useful for debugging)."""
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if (i, j) == self.robot_pos:
                    row += 'R '
                elif self.grid[i][j] == np.inf:
                    row += '# '
                else:
                    row += '. '
            print(row)

def setup_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid World")
    clock = pygame.time.Clock()
    return screen, clock

def draw_grid(world, screen):
    """Render the grid, robot, and goal on the screen."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Determine cell color based on its value
            color = EMPTY_COLOR
            cell_value = world.grid[i][j]
            if cell_value == np.inf:
                color = WALL_COLOR
            elif cell_value == GOLD_REWARD:  # Gold
                color = GOLD_COLOR
            elif cell_value == TRAP_PENALTY:  # Trap
                color = TRAP_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Drawing the grid lines
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT))
        pygame.draw.line(screen, (200, 200, 200), (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE))

    pygame.draw.circle(screen, ROBOT_COLOR, 
                       (int((world.robot_pos[1] + 0.5) * CELL_SIZE), int((world.robot_pos[0] + 0.5) * CELL_SIZE)), 
                       int(CELL_SIZE/3))

    pygame.draw.circle(screen, GOAL_COLOR, 
                       (int((world.goal[1] + 0.5) * CELL_SIZE), int((world.goal[0] + 0.5) * CELL_SIZE)), 
                       int(CELL_SIZE/3))


def main():
    """Main loop"""
    screen, clock = setup_pygame()
    world = GridWorld()
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Move robot based on arrow key press
                if event.key == pygame.K_UP:
                    world.move("up")
                if event.key == pygame.K_DOWN:
                    world.move("down")
                if event.key == pygame.K_LEFT:
                    world.move("left")
                if event.key == pygame.K_RIGHT:
                    world.move("right")
                # Print the score after the move
                print(f"Current Score: {world.score}")
                # Check if the robot reached the goal
                if world.robot_pos == world.goal:
                    print("Robot reached the goal!")
                    print(f"Final Score: {world.score}")
                    running = False
                    break
        # Rendering
        screen.fill(EMPTY_COLOR)
        draw_grid(world, screen)
        pygame.display.flip()

        clock.tick(10)  # FPS

    pygame.quit()

if __name__ == "__main__":
    main()
