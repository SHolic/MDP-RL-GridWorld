# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2 Problem Domain

This code provides a basic and interactive grid world environment where a robot can navigate using the arrow keys. The robot encounters walls that block movement, gold that gives positive rewards, and traps that give negative rewards. The game ends when the robot reaches its goal. The robot's score reflects the rewards it collects and penalties it incurs.

"""

import pygame
import numpy as np
import random
# import warnings

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
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
GOAL_REWARD = 200
LIVING_PENALTY = -1
GAMMA = 0.9
CONVERGE_THRESHOLD = 0.0001

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
        # Setting reward for the goal state
        self.grid[self.goal[0]][self.goal[1]] = GOAL_REWARD

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

    def move(self, direction, step_count, gamma=GAMMA):
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
        reward = (gamma ** step_count) * self.grid[x][y] + LIVING_PENALTY  # step penalty
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
            
class MDPAgent:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.actions = ACTIONS
        self.values = np.zeros((gridworld.size, gridworld.size))
        
    def _get_expected_values(self, i, j, gamma=GAMMA):
        expected_values = []
        
        # Probabilities
        main_prob = 0.8
        other_prob = 0.2 / 3
        
        for action in self.actions:
            total_expected_value = 0
            for prob, each_action in [(main_prob, action)] + [(other_prob, a) for a in self.actions if a != action]:
                ni, nj = i + each_action[0], j + each_action[1]
                
                # Ensure the agent stays within the grid boundaries
                ni = max(0, min(self.gridworld.size - 1, ni))
                nj = max(0, min(self.gridworld.size - 1, nj))
                
                # Ensure the agent don't cross the wall
                if self.gridworld.grid[ni, nj] == np.inf:
                    ni, nj = i, j
                reward = self.gridworld.grid[ni, nj]
                total_expected_value += prob * (reward + LIVING_PENALTY + gamma * self.values[ni, nj])
                
            expected_values.append(total_expected_value)
        return expected_values
    
    def value_iteration(self, gamma=GAMMA, threshold=CONVERGE_THRESHOLD):
        iterations = 0
        while True:
            delta = 0
            for i in range(self.gridworld.size):
                for j in range(self.gridworld.size):
                    # goal state continue
                    if (i, j) == self.gridworld.goal:
                        self.values[i, j] = GOAL_REWARD
                        continue
                    # wall continue
                    if self.gridworld.grid[i, j] == np.inf:
                        continue
                    v = self.values[i, j]
                    self.values[i, j] = max(self._get_expected_values(i, j, gamma))
                    # Check converge
                    delta = max(delta, abs(v - self.values[i, j]))
                    # print(self.values)
            # print(delta)
            iterations += 1
            if delta < threshold:
                break

        # Derive policy from value function
        policy = np.zeros((self.gridworld.size, self.gridworld.size), dtype=tuple)
        for i in range(self.gridworld.size):
            for j in range(self.gridworld.size):
                best_action_index = np.argmax(self._get_expected_values(i, j, gamma))
                policy[i, j] = self.actions[best_action_index]
                
        return self.values, policy, iterations
      
class QLearningAgent:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.actions = ACTIONS
        self.q_table = np.zeros((self.gridworld.size, self.gridworld.size, len(self.actions))) 
        
    def epsilon_greedy(self, i, j, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.q_table[i, j])
    
    def _get_q_table(self, i, j, action_index, action, alpha, gamma=GAMMA):        
        ni, nj = i + action[0], j + action[1]

        # Ensure the agent stays within the grid
        ni = max(0, min(self.gridworld.size - 1, ni))
        nj = max(0, min(self.gridworld.size - 1, nj))
        
        # Ensure the agent don't cross the wall
        if self.gridworld.grid[ni, nj] == np.inf:
            ni, nj = i, j

        reward = self.gridworld.grid[ni, nj]   
        # Q-learning update rule:
        # 1. Calculate the sample using the reward and the maximum Q-value of the next state
        sample = reward + LIVING_PENALTY + gamma * np.max(self.q_table[ni, nj])

        # 2. Update Q-value using the Q-learning update rule (weighted average)
        q = (1 - alpha) * self.q_table[i, j, action_index] + alpha * sample
        return q           
    
    def q_learning(self, alpha=0.1, gamma=GAMMA, epsilon=0.2, threshold=CONVERGE_THRESHOLD):
        iterations = 0
        while True:
            delta = 0
            for i in range(self.gridworld.size):
                for j in range(self.gridworld.size):                    
                    # Use epsilon greedy to choose action
                    action_index = self.epsilon_greedy(i, j, epsilon)
                    action = self.actions[action_index]                                        
                    # goal state continue
                    if (i, j) == self.gridworld.goal:
                        self.q_table[i, j, action_index] = GOAL_REWARD
                        continue
                    # wall continue
                    if self.gridworld.grid[i, j] == np.inf:
                        continue
                    # Update q value
                    old_q_table = self.q_table[i, j, action_index]
                    self.q_table[i, j, action_index] = self._get_q_table(i, j, action_index, action, alpha, gamma)
                    delta = max(delta, abs(old_q_table - self.q_table[i, j, action_index]))
            # print(delta)
            iterations += 1
            if delta < threshold:
                break
    
        # Derive policy from value function
        policy = np.zeros((self.gridworld.size, self.gridworld.size), dtype=tuple)
        for i in range(self.gridworld.size):
            for j in range(self.gridworld.size):
                best_action_index = np.argmax(self.q_table[i, j])
                policy[i, j] = self.actions[best_action_index]
                            
        return self.q_table, policy, iterations
    
    # def q_learning(self, alpha=0.1, gamma=GAMMA, epsilon=0.2, threshold=CONVERGE_THRESHOLD):            
    #     iterations = 0
    #     while True:
    #         delta = 0
    #         for i in range(self.gridworld.size):
    #             for j in range(self.gridworld.size):                    
                    
    #                 # store old value
    #                 old_q_table = self.q_table[i, j]
    #                 # wall continue
    #                 if self.gridworld.grid[i, j] == np.inf:
    #                     continue
                    
    #                 # update q-value
    #                 for action_index, action in enumerate(self.actions):
    #                     ni, nj = i + action[0], j + action[1]
            
    #                     # Ensure the agent stays within the grid
    #                     ni = max(0, min(self.gridworld.size - 1, ni))
    #                     nj = max(0, min(self.gridworld.size - 1, nj))
                
    #                     # goal state continue
    #                     if (i, j) == self.gridworld.goal:
    #                         self.q_table[i, j, action_index] = GOAL_REWARD
    #                         continue
                        
    #                     # Explore with probability epsilon or exploit with probability 1 - epsilon
    #                     if np.random.rand() < epsilon:
    #                         action_index = np.random.randint(0, len(self.actions))
    #                         action = self.actions[action_index]            
            
    #                 reward = self.gridworld.grid[ni, nj]   
    #                 # Q-learning update rule:
    #                 # 1. Calculate the sample using the reward and the maximum Q-value of the next state
    #                 sample = reward + LIVING_PENALTY + gamma * np.max(self.q_table[ni, nj])
            
    #                 # 2. Update Q-value using the Q-learning update rule (weighted average)
    #                 self.q_table[i, j, action_index] = (1 - alpha) * self.q_table[i, j, action_index] + alpha * sample
                    
    #                 # Check convergence
    #                 delta = max(delta, np.max(abs(old_q_table - self.q_table[i, j])))
    #         # print(delta)
    #         iterations += 1
    #         if delta < threshold:
    #             break            

    #     # Derive policy from value function
    #     policy = np.zeros((self.gridworld.size, self.gridworld.size), dtype=tuple)
    #     for i in range(self.gridworld.size):
    #         for j in range(self.gridworld.size):
    #             best_action_index = np.argmax(self.q_table[i, j])
    #             policy[i, j] = self.actions[best_action_index]
                            
    #     return self.q_table, policy, iteration

        
# Convert numeric policy to direction symbols
def policy_display(final_policy):
    directions_map = {0: "→", 1: "←", 2: "↓", 3: "↑"}
    policy_index = np.vectorize(ACTIONS.index)(final_policy)
    return np.vectorize(directions_map.get)(policy_index)

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
    
def execute_policy(policy, world, screen="_", clock="_", delay=500, vis=False):
    # Move the robot based on the provided policy until it reaches the goal.
    # param vis is True when visualizing the step; vis is False make it easier to attain the useful output of performance
    step = 0
    # Give robot the chance to travel all the grids. If still not reach the goal, it must be stuck. 
    max_step = world.size * world.size
    while step < max_step:
        if world.robot_pos == world.goal:
            if vis:
                print("Robot reached the goal!")
                print(f"Final Score: {world.score}")
            return world.score
        
        action = policy[world.robot_pos[0], world.robot_pos[1]]
        direction_map = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}
        world.move(direction_map[action], step)
        step += 1
        
        if vis:
            # Print the score after the move
            print(f"Current Score: {world.score}")
            
            # Rendering
            screen.fill((255, 255, 255))  # Assuming EMPTY_COLOR = (255, 255, 255)
            draw_grid(world, screen)
            # (Drawing grid and other elements is not shown here for brevity)
            pygame.display.flip()
            
            pygame.time.wait(delay)  # Adding delay after each action
            clock.tick(10)  # FPS
    if vis:
        print("The method can't converge within time")
    return np.nan

def main():
    """Modified main loop to execute the optimal policy based on user's choice of agent."""
    # Tell the user provided an argument for agent choice    
    method = input("Please specify the method: 1 for MDP value iteration, 2 for Q-learning: ")
    method = int(method)
    
    if method == 1:
        pass
    elif method == 2:
        alpha = input("Enter the learning rate alpha (default is 0.1): ")
        alpha = float(alpha) if alpha else 0.1
       
        epsilon = input("Enter the exploration probability epsilon (default is 0.2): ")
        epsilon = float(epsilon) if epsilon else 0.2
    else:
        print("Invalid method chosen. Please specify 1 for MDP value iteration or 2 for Q-learning.")
        return
        
    world = GridWorld()
    if method == 1:
        agent = MDPAgent(world)
        print("Computing optimal policy using MDPAgent...")
    elif method == 2:
        agent = QLearningAgent(world)
        print("Computing optimal policy using QLearningAgent...")
    else:
        print("Invalid choice!")
        return  

    # Compute the optimal policy using the chosen agent
    # For MDPAgent, we use value_iteration. For QLearningAgent, we use q_learning.
    _, policy, iterations = agent.value_iteration() if method == 1 else agent.q_learning(alpha=alpha, epsilon=epsilon)
    
    # Execute the computed policy
    screen, clock = setup_pygame()
    execute_policy(policy, world, screen, clock, vis=True)
    print(f"Iterations to converge: {iterations}")
    
    pygame.quit()

if __name__ == "__main__":
    main()
