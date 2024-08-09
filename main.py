import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

## utils.py
def is_list_of_tuples(variable):
    # Check if the variable is a list
    if isinstance(variable, list):
        # Check if all elements in the list are tuples
        return all(isinstance(item, tuple) for item in variable)
    return False

def is_list_of_strings(variable):
    # Check if variable is a list
    if isinstance(variable, list):
        # Check if all elements in the list are strings
        return all(isinstance(item, str) for item in variable)
    return False


## POMDPGridWorldEnv.py
class POMDPGridWorldEnv(gym.Env):
    def __init__(self, cue_1_loc=(2, 0), cue_2='L1', cue_2_locations=[(0, 2), (1, 3), (3, 3), (4, 2)], reward_locations=[(1, 5), (3, 5)]):
        super(POMDPGridWorldEnv, self).__init__()

        self.row = np.random.randint(10)
        self.collumn = np.random.randint(10)
        self.cue_1_loc = cue_1_loc
        self.cue_2_name = cue_2
        self.cue_2_loc_names = ['L1', 'L2', 'L3', 'L4']

        if is_list_of_tuples(cue_2_locations) and len(cue_2_locations) == 4:
            self.cue_2_locations = cue_2_locations
        else:
            self.cue_2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]    

        self.action_space = spaces.Discrete(5)  # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row, self.collumn), dtype=np.float32)

        self.done = False
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
        self.agent_pos = np.random.randint(self.row), np.random.randint(self.collumn)
        # Define a goal position
        self.goal_pos = np.random.randint(self.row), np.random.randint(self.collumn)
        self.done = False
        self.path = [tuple(self.agent_pos)]  # Reset path and include the starting location
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("Environment is done. Please reset it.")

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
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)  # Blue agent

        # Draw the goal
        goal_rect = pygame.Rect(self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)  # Red goal

        # Draw cue 1 with "C1" label
        cue_1_rect = pygame.Rect(self.cue_1_loc[1] * cell_size, self.cue_1_loc[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), cue_1_rect)
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render("C1", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=cue_1_rect.center)
        self.screen.blit(text_surface, text_rect)

        # Draw cue 2 locations with their names
        for idx, loc in enumerate(self.cue_2_locations):
            cue_2_rect = pygame.Rect(loc[1] * cell_size, loc[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (0, 255, 255), cue_2_rect)
            cue_2_label = self.cue_2_loc_names[idx]
            text_surface = font.render(cue_2_label, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=cue_2_rect.center)
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

# Register environment
from gymnasium.envs.registration import register

register(
    id='POMDPGridWorldEnv-v0',
    entry_point='main:POMDPGridWorldEnv',
)


# LangChain setup
prompt_template = PromptTemplate(
    input_variables=["state"],
    template="Given the state of the environment: {state}, what action should the agent take?",
)

llm = OpenAI(model_name="text-davinci-003", openai_api_key="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")

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

class LangChainAgent:
    def __init__(self):
        self.chain = llm_chain

    def decide_action(self, state):
        action = self.chain.run({"state": state})
        return action

langchain_agent = LangChainAgent()

# Test environment
def run_environment():
    # Create the environment
    env = gym.make('POMDPGridWorldEnv-v0')

    observation = env.reset()
    done = False

    while not done:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = event.pos
                    if env.grid_size // 2 - 50 <= mouse_pos[0] <= env.grid_size // 2 + 50 and env.grid_size + 10 <= mouse_pos[1] <= env.grid_size + 40:
                        action = env.action_space.sample()  # Random action
                        action = langchain_agent.decide_action(str(observation))  # Use LangChain agent to decide action
                        observation, reward, done, info = env.step(action)
                        print(f"Observation: {observation}, Action: {action}, Reward: {reward}, Done: {done}")

    env.close()


# LLM Agent
import os
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define a simple prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)

# Initialize the OpenAI LLM with your API key  
llm = OpenAI(model="gpt-3.5-turbo-instruct-0914")  # Specify the OpenAI model you want to use

# Create an LLMChain with the prompt and the LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Define a function to get a response from the LLMChain
def get_response(question):
    response = llm_chain.run({"question": question})
    return response

# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    response = get_response(question)
    print("Response:", response)


if __name__ == "__main__":
    run_environment()
