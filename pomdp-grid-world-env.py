## POMDPGridWorldEnv.py
import numpy as np
import pygame
from gymnasium import spaces
from utils import is_list_of_tuples
import gymnasium as gym
import random

class POMDPGridWorldEnv(gym.Env):
    def __init__(self, is_using_llm=True, cue_1_loc=(2, 0), cue_2='L1', cue_2_locations=[(0, 2), (1, 3), (3, 3), (4, 2)], reward_locations=[(1, 5), (3, 5)]):
        super(POMDPGridWorldEnv, self).__init__()

        self.row = np.random.randint(6, 10)
        self.collumn = np.random.randint(6, 10)
        self.cue_1_loc = cue_1_loc
        self.cue_2_name = cue_2
        self.cue_2_loc_names = ['L1', 'L2', 'L3', 'L4']

        if is_list_of_tuples(cue_2_locations) and len(cue_2_locations) == 4:
            self.cue_2_locations = cue_2_locations
        else:
            self.cue_2_locations = {"L1": (0, 2), "L2":(1, 3), "L3": (3, 3), "L4": (4, 2)}    

        self.action_space = spaces.Discrete(5)  # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row, self.collumn), dtype=np.float32)

        self.done = False
        self.is_using_llm = is_using_llm
        self.grid_world_dimension = (self.row, self.collumn)

        # Initialize pygame
        pygame.init()
        self.grid_size = 600
        self.button_height = 50
        self.screen_size = self.grid_size + self.button_height
        self.cell_size = self.grid_size // max(self.row, self.collumn)
        self.screen = pygame.display.set_mode((self.grid_size, self.screen_size))
        pygame.display.set_caption("POMDP Grid World")

        # Define parameters for the POMDP
        self.reset()

    def reset(self):
        # Initialize the agent's position randomly
        self.agent_pos = np.random.randint(0, self.row), np.random.randint(0, self.collumn)

        if random.choice([True, False]):
            self.reward_conditions = ['LEFT', 'RIGHT']
            self.reward_locations={'LEFT': (2, 2), 'RIGHT': (2, 4)}
        else:
            self.reward_conditions = ['TOP', 'BOTTOM']
            self.reward_locations={'TOP': (1, 5), 'BOTTOM': (3, 5)}

        random_reward = np.random.randint(0, 2)
        
        # Define a goal position
        self.goal_pos = self.reward_locations[self.reward_conditions[random_reward]]
        self.done = False
        self.path = [tuple(self.agent_pos)]  # Reset path and include the starting location
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("Environment is done. Please reset it.")
        
        # if self.is_using_llm:

        # Define the movement
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Right
            self.agent_pos = (self.agent_pos[0], min(self.collumn - 1, self.agent_pos[1] + 1))
        elif action == 2:  # Down
            self.agent_pos = (min(self.row - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 3:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 4:  # Stay
            pass  # No change in position

        if tuple(self.agent_pos) not in self.path:
            self.path.append(tuple(self.agent_pos))  # Add new position to the path

        # Check if the agent has reached the goal
        if self.agent_pos == self.goal_pos:
            reward = 1
            self.done = True
        else:
            reward = 0

        # Get the new observation
        observation = self._get_observation()
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # White background

        # Calculate cell size based on rows and columns
        cell_size = self.grid_size // max(self.row, self.collumn)

        font = pygame.font.SysFont(None, 24)

        # Draw the grid
        for x in range(0, self.grid_size, cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.grid_size))
        for y in range(0, self.grid_size, cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.grid_size, y))

        # Draw the path
        for pos in self.path:
            path_rect = pygame.Rect(pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (192, 192, 192), path_rect)  # Gray color for the path

        # Draw the agent
        agent_rect = pygame.Rect(self.agent_pos[1] * cell_size, self.agent_pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (79, 77, 184), agent_rect)  # Blue agent
        agent_label = 'A'
        agent_text_surface = font.render(agent_label, True, (0, 0, 0))
        agent_text_rect = agent_text_surface.get_rect(center=agent_rect.center)
        self.screen.blit(agent_text_surface, agent_text_rect)

        # Draw the goal
        goal_rect = pygame.Rect(self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)  # Red goal

        # Draw cue 1 with "C1" label
        cue_1_rect = pygame.Rect(self.cue_1_loc[1] * cell_size, self.cue_1_loc[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (227, 81, 23), cue_1_rect)
        text_surface = font.render("C1", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=cue_1_rect.center)
        self.screen.blit(text_surface, text_rect)

        # Draw cue 2 locations with their names
        for idx, loc in enumerate(self.cue_2_locations):
            cue_2_rect = pygame.Rect(loc[1] * cell_size, loc[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (23, 173, 227), cue_2_rect)
            cue_2_label = self.cue_2_loc_names[idx]
            text_surface = font.render(cue_2_label, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=cue_2_rect.center)
            self.screen.blit(text_surface, text_rect)

        # Draw reward locations with their names
        for idx, loc in enumerate(self.reward_locations):
            reward_rect = pygame.Rect(self.reward_locations[loc][1] * cell_size, self.reward_locations[loc][0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (232, 65, 65), reward_rect)
            condition_label = self.reward_conditions[idx]
            text_surface = font.render(condition_label, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=reward_rect.center)
            self.screen.blit(text_surface, text_rect)    

        # Draw the "Continue" button
        button_rect = pygame.Rect(self.grid_size // 2 - 50, self.grid_size + 10, 100, 30)
        pygame.draw.rect(self.screen, (0, 200, 0), button_rect)
        text_surface = font.render("Continue", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def close(self):
        pass

    def _get_observation(self):
        # Observation is a noisy version of the grid state (partially observable)
        noise = np.random.normal(0, 0.1, (self.row, self.collumn))
        grid = np.zeros((self.row, self.collumn))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 0.5
        return np.clip(grid + noise, 0, 1)
