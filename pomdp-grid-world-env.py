## POMDPGridWorldEnv.py
import numpy as np
import pygame
from gymnasium import spaces
from utils import is_list_of_tuples
import gymnasium as gym
import random
import sys

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BUTTON_WIDTH, BUTTON_HEIGHT = 80, 30
BUTTON_SPACING = 10

# Button class
class Button:
    def __init__(self, x, y, width, height, color, label, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.label = label
        self.font = font
        self.text = self.font.render(label, True, (255, 255, 255))
        self.text_rect = self.text.get_rect(center=self.rect.center)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.text, self.text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class POMDPGridWorldEnv(gym.Env):
    def __init__(self, log_file="agent_movement_log.txt",
                is_using_llm=True, start_pos=(0, 0), 
                cue_1_location=(2, 0), 
                cue_2='L1', 
                cue_2_locations=[(0, 2), (1, 3), (3, 3), (4, 2)], 
                reward_conditions = ['TOP', 'BOTTOM'], 
                reward_locations=[(1, 5), (3, 5)], 
                is_random_start = True, 
                is_random_reward = True, 
                is_reward_horizontal = False, 
                is_random_grid = False,
                is_random_reward_locs = False, 
                is_random_cue_2_locs = False):
        super(POMDPGridWorldEnv, self).__init__()

        # Create a clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Random env
        self.is_random_grid = is_random_grid
        self.is_random_reward_locs = is_random_reward_locs
        self.is_random_cue_2_locs = is_random_cue_2_locs

        if is_random_grid:
            self.row = np.random.randint(6, 10)
            self.collumn = np.random.randint(6, 10)
        else:
            self.row = 6
            self.collumn = 8
        
        self.grid_world_dimension = (self.row, self.collumn)

        # Initialize the agent's position randomly
        self.is_random_start = is_random_start

        self.start = start_pos

        if self.is_random_start:
            self.start = np.random.randint(0, self.row), np.random.randint(0, self.collumn)

        # self.agent_pos = self.start
        self.agent_action = 'STAY'

        self.cue_1_location = cue_1_location
        self.cue_1_obs = 'Null'
        self.is_cue_1_reached = False

        self.cue_2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue_2_location = 'Null'
        self.cue_2_name = cue_2
        self.cue_2_obs = 'Null'
        self.is_cue_2_reached = False

        self.is_random_reward = is_random_reward
        self.is_reward_horizontal = is_reward_horizontal
        self.reward_conditions = reward_conditions
        self.reward_locations = reward_locations
        self.reward_location = 'Null'
        self.is_reward_horizontal = False 
        self.reward_obs = 'Null'
        self.prev_reward_location = 'Null'

        if is_list_of_tuples(cue_2_locations) and len(cue_2_locations) == 4:
            self.cue_2_locations = cue_2_locations
        else:
            self.cue_2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

        self.action_space = spaces.Discrete(5)  # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row, self.collumn), dtype=np.float32)

        self.done = False
        self.is_using_llm = is_using_llm

        self.result = 'Null'

        self.environment_setup = {
            "grid_world_dimension": self.grid_world_dimension,
            "start": self.start,
            "cue_1_location": self.cue_1_location
        }

        # Increment the step counter
        self.current_step = 0
        
        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)
        self.grid_size = 700
        self.info_height = 50
        self.screen_size = self.grid_size + 2*self.info_height
        self.cell_size = self.grid_size // max(self.row, self.collumn)
        self.screen = pygame.display.set_mode((self.grid_size, self.screen_size))
        self.cell_size = self.grid_size // max(self.row, self.collumn)
        self.grid_lines = []
        self.cue_2_rects = []
        self.reward_rects = []
        pygame.display.set_caption("POMDP Grid World")


        self.log_file = log_file
        self.reset_log_file(self.log_file)  # Resets the log file at the start of each run
        self.reset_log_file('agent_path.txt')
        # Define parameters for the POMDP
        self.reset()

    def reset_agent_pos(self):
        if self.is_random_start:
            self.start = np.random.randint(0, self.row), np.random.randint(0, self.collumn)
        self.agent_pos = self.start
        self.cue_1_obs = 'Null'
        self.cue_2_location = 'Null'
        self.cue_2_obs = 'Null'
        self.reward_condition = 'Null'
        self.prev_reward_location = self.reward_location
        self.reward_location = 'Null'
        self.is_cue_1_reached = False
        self.is_cue_2_reached = False
        self.current_step = 0  # Reset the step count

    def reset_ui(self):
        self.cue_2_rects = []
        self.reward_rects = []
        self.path = [tuple(self.agent_pos)]  # Reset path and include the starting location

    def reset(self):
        self.done = False
        self.reset_agent_pos()
        self.reset_ui()

        self.reset_log_file('agent_path.txt')

        # Create a static env where the reward conditions are always TOP and BOTTOM at the same location
        self.reward_conditions = ['TOP', 'BOTTOM']
        # Random reward on the same collumn
        self.reward_locations = [(1, 5), (3, 5)]

        if self.is_random_reward:
            self.is_reward_horizontal = random.choice([True, False])
            if self.is_reward_horizontal:
                self.reward_conditions = ['LEFT', 'RIGHT']
                # Random reward on the same row
                self.reward_locations = [(2, 2), (2, 4)]
            else:
                self.reward_conditions = ['TOP', 'BOTTOM']
                # Random reward on the same collumn
                self.reward_locations = [(1, 5), (3, 5)]

        random_reward = np.random.randint(0, 2)
        
        # Define a goal position
        self.goal_pos = self.reward_locations[random_reward]
        return self._get_observation(), {}

    def reset_log_file(self, file_name):
        with open(self.log_file, 'w') as file:
            file.write("\n")
    
    def log_agent_movement(self, step_count, next_action, action_reason, next_position, position, result):
        with open(self.log_file, 'a') as file:
            log_entry = f"Step {step_count}: Position - {position} -  Next action - {next_action}, Next position - {next_position}, Reason - {action_reason} , Result - {self.result}\n"
            file.write(log_entry)

    def log_random_obs(self, step_count, obs, obs_location):
        with open('random_obs.txt', 'a') as file:
            log_entry = f"Step {step_count} Random {obs} at {obs_location}\n"
            file.write(log_entry)

    def log_path(self, step_count, agent_pos):
        with open('agent_path.txt', 'a') as file:
            log_entry = f"Step {step_count}: Agent position {agent_pos}\n"
            file.write(log_entry)

    def log_info(self, info):
        with open('debug_env_info.txt', 'a') as file:
            file.write(f"{info}\n")

    def log_experiment_results(self, info):
        with open('experiments_results.txt', 'a') as file:
            file.write(f"{info}\n")

    def evaluate_result(self):
        # Result: CHEESE - SHOCK - TIMES EXCEED - WRONG NEXT POSITION - 
        result = False

        if self.reward_obs == 'CHEESE':
            result = True 
        
    def step(self, action): 
        print('response: ', action)
        print('infering_times: ', action['infering_times'])
        reset = False

        if action['infering_times'] == 25:
            reset = True

        else:
            self.agent_action = action['next_action']
            self.agent_pos = eval(action['position'])
            
            if self.agent_pos == self.cue_1_location and self.is_cue_1_reached != True:
                self.is_cue_1_reached = True
                self.random_obs('cue_1')

            if self.agent_pos == self.cue_2_location and self.is_cue_1_reached and self.is_cue_2_reached != True:
                self.is_cue_2_reached = True
                self.random_obs('cue_2')

            # Define the movement
            # if self.agent_action == 0 or self.agent_action == 'MOVE_UP':  # Up
            #     self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
            # elif self.agent_action == 1 or self.agent_action == 'MOVE_RIGHT':  # Right
            #     self.agent_pos = (self.agent_pos[0], min(self.collumn - 1, self.agent_pos[1] + 1))
            # elif self.agent_action == 2  or self.agent_action == 'MOVE_DOWN':  # Down
            #     self.agent_pos = (min(self.row - 1, self.agent_pos[0] + 1), self.agent_pos[1])
            # elif self.agent_action == 3  or self.agent_action == 'MOVE_LEFT':  # Left
            #     self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
            # elif self.agent_action == 4  or self.agent_action == 'STAY':  # Stay
            #     pass  # No change in position

            if tuple(self.agent_pos) not in self.path:
                self.path.append(tuple(self.agent_pos))  # Add new position to the path
                self.log_path(self.current_step, self.agent_pos)

            # Check if the agent has reached the goal
            # if self.agent_pos == self.goal_pos:
            #     reward = 1
            #     self.done = True
            # else:
            #     reward = 0

            # @NOTE: here we use the same variable `reward_locations` to create both the agent's generative model (the `A` matrix) as well as the generative process.
            # This is just for simplicity, but it's not necessary -  you could have the agent believe that the Cheese/Shock are actually stored in arbitrary, incorrect locations.
            
            self.reward_obs = 'Null'

            if self.is_cue_2_reached:
                if self.is_reward_horizontal:
                    if self.agent_pos == self.reward_locations[0]:
                        if self.cue_2_obs == 'LEFT':
                            self.reward_obs = 'CHEESE'
                        else:
                            self.reward_obs = 'SHOCK'
                    elif self.agent_pos == self.reward_locations[1]:
                        if self.cue_2_obs == 'RIGHT':
                            self.reward_obs = 'CHEESE'
                        else:
                            self.reward_obs = 'SHOCK'
                else:
                    if self.agent_pos == self.reward_locations[0]:

                        if self.cue_2_obs == 'TOP':
                            self.reward_obs = 'CHEESE'
                        else:
                            self.reward_obs = 'SHOCK'
                    elif self.agent_pos == self.reward_locations[1]:
                        if self.cue_2_obs == 'BOTTOM':
                            self.reward_obs = 'CHEESE'
                        else:
                            self.reward_obs = 'SHOCK'

            if self.reward_obs != 'Null':
                reset = True

            # current_location = self.agent_pos
            # cue_1_obs = self.cue_1_obs
            # cue_2_obs = self.cue_2_obs

            # Get the new observation
        observation = self._get_observation()

        # Log the agent's movement after the step
        self.log_agent_movement(self.current_step, next_position = action['next_position'], next_action = action['next_action'], action_reason = action['action_reason'], position = self.agent_pos, result = self.result)
        
        # Increment the step counter
        self.current_step += 1
        return observation, self.reward_obs, self.done, { 'reset': reset }


    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # White background

        # Pre-create and cache grid lines (static grid lines can be cached)
        if len(self.grid_lines) == 0:
            self.grid_lines = []
            for x in range(0, self.grid_size, self.cell_size):
                self.grid_lines.append(((x, 0), (x, self.grid_size)))
            for y in range(0, self.grid_size, self.cell_size):
                self.grid_lines.append(((0, y), (self.grid_size, y)))
                  
        # Draw the grid (from cached grid lines)
        for line in self.grid_lines:
            pygame.draw.line(self.screen, (0, 0, 0), line[0], line[1])

        # Draw the goal
        # goal_rect = pygame.Rect(self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size)
        # pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)  # Red goal

        # Draw cue 1 with "C1" label
        cue_1_rect = pygame.Rect(self.cue_1_location[1] * self.cell_size, self.cue_1_location[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (227, 81, 23), cue_1_rect)
        text_surface = self.font.render("C1", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=cue_1_rect.center)
        self.screen.blit(text_surface, text_rect)

        # Draw cue 2 locations
        if len(self.cue_2_rects) == 0:
            self.cue_2_rects = [pygame.Rect(loc[1] * self.cell_size, loc[0] * self.cell_size, self.cell_size, self.cell_size) for loc in self.cue_2_locations]

        for i, rect in enumerate(self.cue_2_rects):
            pygame.draw.rect(self.screen, (23, 173, 227), rect)
            text_surface = self.font.render(self.cue_2_loc_names[i], True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)

        # Draw reward locations
        if len(self.reward_rects) == 0:
            self.reward_rects = [pygame.Rect(loc[1] * self.cell_size, loc[0] * self.cell_size, self.cell_size, self.cell_size) for loc in self.reward_locations]
            
        for i, rect in enumerate(self.reward_rects):
            pygame.draw.rect(self.screen, (232, 65, 65), rect)
            text_surface = self.font.render(self.reward_conditions[i], True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)

        # Draw the path
        for pos in self.path:
            path_rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (192, 192, 192), path_rect)  # Gray color for the path
            
            displayed_text = ''

            if pos == self.cue_1_location:
                displayed_text  = 'C1'
            if pos in self.cue_2_locations:
                idx = self.cue_2_locations.index(pos)
                displayed_text = self.cue_2_loc_names[idx]
            if pos in self.reward_locations:
                idx = self.reward_locations.index(pos)
                displayed_text = self.reward_conditions[idx]        
            if displayed_text != '':
                text_surface = self.font.render(displayed_text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=path_rect.center)
                self.screen.blit(text_surface, text_rect)
        
        # Draw the agent
        agent_rect = pygame.Rect(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (79, 77, 184), agent_rect)  # Blue agent
        agent_label = 'A'
        agent_text_surface = self.font.render(agent_label, True, (0, 0, 0))
        agent_text_rect = agent_text_surface.get_rect(center=agent_rect.center)
        self.screen.blit(agent_text_surface, agent_text_rect)
        
        # ADD AN INFORMATION LINE AT THE BOTTOM OF THE POP UP
        info_text_surface = self.font.render(f"Current location: {self.agent_pos}, Action: {self.agent_action}, Cue 2: {self.cue_1_obs} - {self.cue_2_location}", True, (0, 0, 0))
        info_x = 10  # Padding from the left edge
        info_y = self.screen_size - self.info_height + 10  # Positioned at the bottom within the info area
        self.screen.blit(info_text_surface, (info_x, info_y))
        
        info_1_text_surface = self.font.render(f"Reward condition: {self.cue_2_obs} - {self.reward_location}", True, (0, 0, 0))
        info_1_x = 10  # Padding from the left edge
        info_1_y = self.screen_size - 1.5*self.info_height + 10 # Positioned at the bottom within the info area
        self.screen.blit(info_1_text_surface, (info_1_x, info_1_y))

        info_2_text_surface_2 = self.font.render(f"Random start: {self.is_random_start}", True, (0, 0, 0))
        info_2_x = 10  # Padding from the left edge
        info_2_y = self.screen_size - 2*self.info_height + 10  # Positioned at the bottom within the info area
        self.screen.blit(info_2_text_surface_2, (info_2_x, info_2_y))
        
        if (self.reward_obs != 'Null'):
            self.show_reward_popup()
            self.done = True
            self.reset()        
        
        pygame.display.flip()

    def random_obs(self, type):
        obs = ''
        obs_location = ''
        if type == 'cue_1':
            rand_idx = np.random.randint(4)
            self.cue_1_obs = self.cue_2_loc_names[rand_idx]
            self.cue_2_location = self.cue_2_locations[rand_idx]
            obs = self.cue_1_obs
            obs_location = self.cue_2_location

        else:
            rand_idx = np.random.randint(2)
            self.cue_2_obs = self.reward_conditions[rand_idx]
            self.reward_location = self.reward_locations[rand_idx]
            obs = self.cue_2_obs
            obs_location = self.reward_location

        self.log_random_obs(self.current_step, obs, obs_location)
        
    def show_reward_popup(self):
        popup_width = 300
        popup_height = 200
        popup_surface = pygame.Surface((popup_width, popup_height))
        popup_surface.fill((111, 173, 128))
        BUTTON_LABELS = self.reward_obs

        # Calculate center position for the popup
        center_x = (self.screen_size - popup_width) // 2
        center_y = (self.screen_size - popup_height)

        text_surface = self.font.render(BUTTON_LABELS, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(popup_width//2, popup_height//2))
        popup_surface.blit(text_surface, text_rect)

        self.screen.blit(popup_surface, (center_x//2, center_y//2))
        pygame.display.flip()

    def close(self):
        pass

    def _get_observation(self):
        # Observation is a noisy version of the grid state (partially observable)
        noise = np.random.normal(0, 0.1, (self.row, self.collumn))
        grid = np.zeros((self.row, self.collumn))
        # grid[self.agent_pos[0], self.agent_pos[1]] = 1
        # grid[self.goal_pos[0], self.goal_pos[1]] = 0.5
        return np.clip(grid + noise, 0, 1)
