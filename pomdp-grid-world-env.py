## POMDPGridWorldEnv.py
import numpy as np
import pygame
from gymnasium import spaces
from utils import is_list_of_tuples
import gymnasium as gym
import random
import sys
import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BUTTON_WIDTH, BUTTON_HEIGHT = 80, 30
BUTTON_SPACING = 10

class POMDPGridWorldEnv(gym.Env):
    def __init__(self, log_file="agent_movement_log.txt",
                is_using_llm=True,
                start_pos=(0, 0), 
                is_random_start = True, 
                is_random_reward = True, 
                is_reward_horizontal = False, 
                is_random_grid = False,
                is_random_cue_1 = False, 
                is_random_cue_2_locs = False,
                step_limitation = 50,
                iteration_limitation = 50,
                type='unseen'):
        
        super(POMDPGridWorldEnv, self).__init__()

        # Create a clock to control the frame rate
        self.clock = pygame.time.Clock()
        self.step_limitation = step_limitation
        self.iteration_limitation = iteration_limitation

        # Random env
        self.type = type
        self.is_random_grid = is_random_grid
        self.is_random_cue_1 = is_random_cue_1
        self.is_random_cue_2_locs = is_random_cue_2_locs
        self.is_random_reward = is_random_reward
        self.is_reward_horizontal = is_reward_horizontal
        self.existed_locations = [] #only for randomize locations
            
        # Cue, reward information
        self.cue_1_obs = 'Null'
        self.is_cue_1_reached = False
        self.cue_2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue_2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]
        self.cue_2_location = 'Null'
        self.cue_2_name = 'L1'
        self.cue_2_obs = 'Null'
        self.is_cue_2_reached = False
        self.reward_conditions = ['FIRST', 'SECOND']
        self.reward_locations = [(1, 5), (3, 5)]
        self.reward_location = 'Null'
        self.is_reward_horizontal = False 
        self.reward_obs = 'Null'
        self.prev_reward_location = 'Null'
        self.step_info = {
                'next_action': '',
                'action_reason': '',
                'position': '',
                'next_position': '',
                'step_limitation': '',
                'iteration_times': ''}

        if is_random_grid:
            self.row = np.random.randint(6, 10)
            self.column = np.random.randint(6, 10)
        else:
            self.row = 6
            self.column = 8
        
        self.grid_dims = [self.row, self.column]
        self.grid_world_dimension = (self.row, self.column)

        # Initialize the agent's position randomly
        self.is_random_start = is_random_start

        self.start = start_pos

        if self.is_random_start:
            self.start = np.random.randint(0, self.row), np.random.randint(0, self.column)
            self.existed_locations.append(self.start)

        self.agent_pos = self.start
        self.environment_setup = {}

        self.agent_action = 'STAY'

        self.cue_1_location = (2, 0)

        if self.is_reward_horizontal:
            self.reward_locations = [(2, 2), (2, 4)]


        self.action_space = spaces.Discrete(5)  # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row, self.column), dtype=np.float32)

        self.done = False
        self.is_using_llm = is_using_llm

        self.result = 'Null'

        # Increment the step counter
        self.current_step = 0
        
        # Increment the iteration counter
        self.current_iteration = 0
        
        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)
        self.grid_size = 700
        self.sidebar_width = 800
        self.info_height = 50
        self.screen_width = 1500
        self.screen_height = self.grid_size + 2*self.info_height
        self.cell_size = self.grid_size // max(self.row, self.column)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.grid_lines = []
        self.cue_2_rects = []
        self.reward_rects = []
        self.text_blocks = []
        self.scroll_y = 0
        pygame.display.set_caption("POMDP Grid World")

        # Result plotting
        self.result = False
        self.sr = 0
        self.gc = 0
        self.total_sr = 0
        self.total_gc = 0
        self.ne = 0
        self.current_result = 0
        self.current_exp_results = [0]
        self.final_position = 'Null'
        
        self.log_file = log_file
        self.reset_log_file(self.log_file)  # Resets the log file at the start of each run
        self.reset_log_file('agent_path.txt')
        self.reset_log_file(f'result_{self.type}.txt')
        # Define parameters for the POMDP
        # self.reset()

    def reset_env_pos(self):
        if self.is_random_start:
            self.start = np.random.randint(0, self.row), np.random.randint(0, self.column)
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
        self.existed_locations = [self.start]

    def reset_ui(self):
        self.cue_2_rects = []
        self.reward_rects = []
        self.text_blocks = []
        self.path = [tuple(self.agent_pos)]  # Reset path and include the starting location

    def reset(self):
        self.done = False
        self.reset_env_pos()
        if self.is_random_cue_1:
            self.cue_1_location = self.random_location_excluding(self.existed_locations)
        if self.is_random_cue_2_locs:
            self.cue_2_locations = self.generate_multiple_locations(len(self.cue_2_loc_names), self.existed_locations)            
        if self.is_random_reward:
            self.is_reward_horizontal = random.choice([True, False])
            self.reward_locations = self.randomize_reward_locations(grid_columns=self.column, grid_rows=self.row , is_reward_horizontal=self.is_reward_horizontal)

        self.environment_setup = {
            "grid_world_dimension": self.grid_world_dimension,
            "start": self.start,
            "cue_1_location": self.cue_1_location
        }

        self.step_info = {
                'next_action': '',
                'action_reason': '',
                'position': '',
                'next_position': '',
                'current_step': '',
                'current_iteration': ''}

        self.reset_ui()
        self.reset_log()

        random_reward = np.random.randint(0, 2)
        
        # Define a goal position
        self.goal_pos = self.reward_locations[random_reward]
        self.current_iteration+=1
        if self.current_iteration == self.iteration_limitation:
            self.log_info(self.current_exp_results, f'result_{self.type}.txt')
        return self._get_observation(), {'current_step': self.current_step, 'current_iteration': self.current_iteration, 'loc_obs': self.agent_pos, 'cue_1_obs': self.cue_1_obs, 'cue_2_obs': self.cue_2_obs, 'reward_obs': self.reward_obs }
    
    def reset_log(self):
        self.reset_log_file('agent_path.txt')

    # Random excluding existed location
    def random_location_excluding(self, excluded_locations):
        while True:
            # Generate a random location
            new_location = (np.random.randint(0, self.row), np.random.randint(0, self.column))
            # Check if it is not in the excluded locations
            if new_location not in excluded_locations:
                return new_location

    # Random multiple location excluding existed location
    def generate_multiple_locations(self, number_of_locations, existed_locations):
        generated_locations = []
        while len(generated_locations) < number_of_locations:  # +1 to include the already added new_location
            new_loc = self.random_location_excluding(existed_locations)
            if new_loc not in existed_locations:
                generated_locations.append(new_loc)
                existed_locations.append(new_loc)
        return generated_locations

    def randomize_reward_locations(self, grid_columns, grid_rows, is_reward_horizontal):
        if is_reward_horizontal:
            # Choose a random row
            random_row = np.random.randint(grid_rows)
            # Randomly choose two different columns and sort them
            cols = np.random.choice(grid_columns, size=2, replace=False)
            col1, col2 = sorted(cols)
            reward_locations = [(random_row, col1), (random_row, col2)]
        else:
            # Choose a random column
            random_column = np.random.randint(grid_columns)
            # Randomly choose two different rows and sort them
            rows = np.random.choice(grid_rows, size=2, replace=False)
            row1, row2 = sorted(rows)
            reward_locations = [(row1, random_column), (row2, random_column)]
        
        return reward_locations

    def reset_log_file(self, file_name):
        with open(self.log_file, 'w') as file:
            file.write("")
    
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

    def log_info(self, info, file_name):
        with open(file_name, 'a') as file:
            file.write(f"{info}")

    def log_experiment_results(self, info):
        with open('experiments_results.txt', 'a') as file:
            file.write(f"{info}\n")

    def evaluate_result(self):
        # Result: CHEESE - SHOCK - TIMES EXCEED - WRONG NEXT POSITION - 
        result = False

        if self.reward_obs == 'CHEESE':
            result = True 
        
    def step(self, action): 
        reset = False
        cue_1_obs = 'Null'
        self.step_info = action
        if self.current_step == self.step_limitation:
            self.reset()
            reset = True

        else:
            self.agent_action = action['next_action']

            if self.is_using_llm:
                self.agent_pos = eval(action['position'])
            else:
                (Y, X) = self.agent_pos

                if self.agent_action == "UP":

                    Y_new = Y - 1 if Y > 0 else Y
                    X_new = X

                elif self.agent_action == "DOWN":

                    Y_new = Y + 1 if Y < (self.grid_dims[0]-1) else Y
                    X_new = X

                elif self.agent_action == "LEFT":
                    Y_new = Y
                    X_new = X - 1 if X > 0 else X

                elif self.agent_action == "RIGHT":
                    Y_new = Y
                    X_new = X +1 if X < (self.grid_dims[1]-1) else X

                elif self.agent_action == "STAY":
                    Y_new, X_new = Y, X

                self.agent_pos = (Y_new, X_new) # store the new grid location

            if self.agent_pos == self.cue_1_location and self.is_cue_1_reached != True:
                self.is_cue_1_reached = True
                self.current_result = 1
                self.random_obs('cue_1')

            if self.agent_pos == self.cue_2_location and self.is_cue_1_reached and self.is_cue_2_reached != True:
                self.is_cue_2_reached = True
                self.current_result = 2
                self.random_obs('cue_2')

            if tuple(self.agent_pos) not in self.path:
                self.path.append(tuple(self.agent_pos))  # Add new position to the path
                self.log_path(self.current_step, self.agent_pos)
                self.log_path('self.path: ', self.path)
            
            self.reward_obs = 'Null'

            if self.is_cue_2_reached:
                if self.agent_pos == self.reward_locations[0]:
                    if self.cue_2_obs == 'FIRST':
                        self.reward_obs = 'CHEESE'
                    else:
                        self.reward_obs = 'SHOCK'
                elif self.agent_pos == self.reward_locations[1]:
                    if self.cue_2_obs == 'SECOND':
                        self.reward_obs = 'CHEESE'
                    else:
                        self.reward_obs = 'SHOCK'


            if self.reward_obs != 'Null':
                if self.reward_obs == 'CHEESE':
                    self.current_result = 4
                if self.reward_obs == 'SHOCK':
                    self.current_result = 3

                reset = True
        
        if reset == True:
            self.current_exp_results.append(self.current_result)
        # Get the new observation
        observation = self._get_observation()

        # Log the agent's movement after the step
        if self.is_using_llm:
            self.log_agent_movement(self.current_step, next_position = action['next_position'], next_action = action['next_action'], action_reason = action['action_reason'], position = self.agent_pos, result = self.result)
        
        # Increment the step counter
        self.current_step += 1
        return observation, self.reward_obs, self.done, { 'reset': reset, 'current_step': self.current_step, 'current_iteration': self.current_iteration, 'loc_obs': self.agent_pos, 'cue_1_obs': self.cue_1_obs, 'cue_2_obs': self.cue_2_obs, 'reward_obs': self.reward_obs  }

    # Render screen
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
        
        info_text_surface = self.font.render(f"Step #{self.current_step}/{self.step_limitation}, Current location: {self.agent_pos}, Action: {self.agent_action}, Cue 2: {self.cue_1_obs} - {self.cue_2_location}", True, (0, 0, 0))
        info_x = 10  # Padding from the left edge
        info_y = self.screen_height - self.info_height + 10  # Positioned at the bottom within the info area
        self.screen.blit(info_text_surface, (info_x, info_y))
        
        info_1_text_surface = self.font.render(f"Reward condition: {self.cue_2_obs} - {self.reward_location}", True, (0, 0, 0))
        info_1_x = 10  # Padding from the left edge
        info_1_y = self.screen_height - 1.5*self.info_height + 10 # Positioned at the bottom within the info area
        self.screen.blit(info_1_text_surface, (info_1_x, info_1_y))

        info_2_text_surface_2 = self.font.render(f"Random start: {self.is_random_start}", True, (0, 0, 0))
        info_2_x = 10  # Padding from the left edge
        info_2_y = self.screen_height - 2*self.info_height + 10  # Positioned at the bottom within the info area
        self.screen.blit(info_2_text_surface_2, (info_2_x, info_2_y))

        # Fill the main screen and sidebar area
        sidebar_color = (255, 255, 255)  # Light grey sidebar
        sidebar_rect = pygame.Rect(self.screen_width - self.sidebar_width, 0, self.sidebar_width, self.screen_height)
        self.screen.fill(sidebar_color, sidebar_rect)
        self.draw_sidebar(self.screen, sidebar_rect, sidebar_color)
        
        if (self.reward_obs != 'Null'):
            self.show_reward_popup()
            self.done = True
            self.reset()
        
        self.update_display()

    def draw_sidebar(self, screen, sidebar_rect, sidebar_color):
        # Fill sidebar
        screen.fill(sidebar_color, sidebar_rect)
        plot_position = (sidebar_rect.x, 50)  # Adjust as needed
        # Draw text in the sidebar with scrolling
        self.draw_plot_on_pygame(self.screen, plot_position, np.arange(0, len(self.current_exp_results)), self.current_exp_results)

    def draw_plot_on_pygame(self, screen, plot_position, x_values = np.arange(0, 51), y_values = np.arange(0, 51)):
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize to make the plot bigger (width=8, height=6)

        # Plotting the custom values
        ax.plot(x_values, y_values, marker='o', linestyle='-', color='b')  # Adjust marker and line style as needed
        
        # Add text on top of the plot
        ax.text(0.5, 1.05, f'Experiment #{self.current_iteration}/{self.iteration_limitation}', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes, fontsize=14, fontweight='bold')

        # Set the labels for the axes
        ax.set_xlabel('Experiments')
        ax.set_ylabel('Results')
        
        # Set custom labels for the Y-axis ticks
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels([
            'Cue 1',
            'Cue 2',
            'Shock',
            'Cheese'
        ])

        # Ensure X-axis has integer values only
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # Ensure X-axis has continuous integer values only greater than 0
        ax.set_xlim(left=0)  # Ensure X-axis starts from 0 to show the root (origin)

        # Ensure grid lines or ticks are correctly placed
        ax.yaxis.set_ticks_position('left')  # Show ticks only on the left spine
        ax.xaxis.set_ticks_position('bottom')  # Show ticks only on the bottom spine

        # Render the plot to a Pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        # Create a Pygame surface
        size = canvas.get_width_height()
        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

        # Blit the plot surface onto the Pygame screen
        screen.blit(plot_surface, plot_position)
        plt.close(fig)  # Close the figure after drawing to avoid memory issues

    def update_display(self):
        pygame.display.flip()

    def random_obs(self, type):
        obs = 'Null'
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

        return obs
        
    def show_reward_popup(self):
        popup_width = 300
        popup_height = 200
        popup_surface = pygame.Surface((popup_width, popup_height))
        popup_surface.fill((111, 173, 128))
        BUTTON_LABELS = self.reward_obs

        # Calculate center position for the popup
        center_x = (self.screen_height - popup_width) // 2
        center_y = (self.screen_height - popup_height)

        text_surface = self.font.render(BUTTON_LABELS, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(popup_width//2, popup_height//2))
        popup_surface.blit(text_surface, text_rect)

        self.screen.blit(popup_surface, (center_x//2, center_y//2))
        pygame.display.flip()

    def close(self):
        pass

    def _get_observation(self):
        # Observation is a noisy version of the grid state (partially observable)
        noise = np.random.normal(0, 0.1, (self.row, self.column))
        grid = np.zeros((self.row, self.column))
        # grid[self.agent_pos[0], self.agent_pos[1]] = 1
        # grid[self.goal_pos[0], self.goal_pos[1]] = 0.5
        return np.clip(grid + noise, 0, 1)
