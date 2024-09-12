import gymnasium as gym
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from gymnasium.envs.registration import register
import pygame
import os
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from typing import List
from LLMAgent import LLMAgent
from utils import parse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from pymdp.agent import Agent
from pymdp import utils, maths
import numpy as np

def reset_modalities():
    

# Test environment
def run_environment():
        
    # Register environment
    register(
        id='POMDPGridWorldEnv-v0',
        entry_point='pomdp-grid-world-env:POMDPGridWorldEnv',
    )

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("Please set the OpenAI API key in the .env file.")

    # Use the OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # LangChain setup
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", model_kwargs={ "response_format": { "type": "json_object" }})
    current_iteration = 0
    step_limitation = 20
    iteration_limitation = 50
    reset = False

    # Create the environment
    env = gym.make('POMDPGridWorldEnv-v0',
                #    is_random_grid = True,
                #    is_random_start = True,
                #    is_random_reward = True,
                #    is_random_cue_1 = True,
                #    is_random_cue_2_locs = True,
                   is_random_start = False,
                   is_random_reward = False,
                   step_limitation = step_limitation,
                   iteration_limitation = iteration_limitation,
                   is_using_llm = False)
    
    observation, info = env.reset()
    
    env.render()

    done = False

    # Define Environment and Agent Parameters
    grid_dims = [env.row, env.column] # dimensions of the grid (number of rows, number of columns)
    num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)
    num_cue_2_locations = len(env.cue_2_loc_names)  # Number of possible locations for Cue 2
    num_reward_conditions = len(env.reward_conditions)  # Reward conditions: "TOP" or "BOTTOM"

    # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates
    grid = np.arange(num_grid_points).reshape([env.row, env.column])
    it = np.nditer(grid, flags=["multi_index"])

    loc_list = []
    while not it.finished:
        loc_list.append(it.multi_index)
        it.iternext()

    # Define locations for Cue 1 and Cue 2
    cue_1_location = env.cue_1_location
    cue_2_location = env.cue_2_location
    reward_locations = env.reward_locations

    # Initialize Observation Matrices: A
    num_states = [num_grid_points, len(env.cue_2_locations), len(env.reward_conditions)]
    cue_1_names = ['Null'] + env.cue_2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
    cue_2_names = ['Null'] + env.reward_conditions
    reward_names = ['Null', 'CHEESE', 'SHOCK']
    num_obs = [num_grid_points, len(cue_1_names), len(cue_2_names), len(reward_names)]

    # Observation array model: A array
    A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs] # list of shapes of modality-specific A[m] arrays
    A = np.empty(len(A_m_shapes), dtype=object) # initialize A array to an object array of all-zero subarrays
    
    # Fill A with all-zero subarrays of the shapes specified in A_m_shapes
    for i, shape in enumerate(A_m_shapes):
        A[i] = np.zeros(shape)

    # make the location observation only depend on the location state (proprioceptive observation modality)
    A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, num_states[1], num_states[2]))

    # make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
    A[1][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

    # Make the Cue 1 signal depend on 1) being at the Cue 1 location an) the location of Cue 2
    for i, cue_loc_2_i in enumerate(env.cue_2_locations):
        A[1][0,loc_list.index(cue_1_location),i,:] = 0.0
        A[1][i+1,loc_list.index(cue_1_location),i,:] = 1.0

    # make the reward observation depend on the location (being at reward location) and the reward condition
    A[3][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

    rew_1st_idx = loc_list.index(reward_locations[0]) # linear index of the location of the "FIRST" reward location
    rew_2nd_idx = loc_list.index(reward_locations[1]) # linear index of the location of the "SECOND" reward location

    # fill out the contingencies when the agent is in the "FIRST" reward location
    A[3][0,rew_1st_idx,:,:] = 0.0
    A[3][1,rew_1st_idx,:,0] = 1.0
    A[3][2,rew_1st_idx,:,1] = 1.0

    # fill out the contingencies when the agent is in the "SECOND" reward location
    A[3][0,rew_2nd_idx,:,:] = 0.0
    A[3][1,rew_2nd_idx,:,1] = 1.0
    A[3][2,rew_2nd_idx,:,0] = 1.0

    # make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
    A[2][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

    for i, cue_loc2_i in enumerate(env.cue_2_locations):

        # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
        A[2][0,loc_list.index(cue_loc2_i),i,:] = 0.0
        A[2][1,loc_list.index(cue_loc2_i),i,0] = 1.0
        A[2][2,loc_list.index(cue_loc2_i),i,1] = 1.0

    # make the reward observation depend on the location (being at reward location) and the reward condition
    A[3][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

    rew_top_idx = loc_list.index(reward_locations[0]) # linear index of the location of the "FIRST" reward location
    rew_bott_idx = loc_list.index(reward_locations[1]) # linear index of the location of the "SECOND" reward location

    # fill out the contingencies when the agent is in the "FIRST" reward location
    A[3][0,rew_top_idx,:,:] = 0.0
    A[3][1,rew_top_idx,:,0] = 1.0
    A[3][2,rew_top_idx,:,1] = 1.0

    # fill out the contingencies when the agent is in the "SECOND" reward location
    A[3][0,rew_bott_idx,:,:] = 0.0
    A[3][1,rew_bott_idx,:,1] = 1.0
    A[3][2,rew_bott_idx,:,0] = 1.0

    # The transition model: B array
    # initialize `num_controls`
    num_controls = [5, 1, 1]

    # initialize the shapes of each sub-array `B[f]`
    B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

    # create the `B` array and fill it out
    B = utils.obj_array_zeros(B_f_shapes)

    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    # fill out `B[0]` using the
    for action_id, action_label in enumerate(actions):

        for curr_state, grid_location in enumerate(loc_list):

            y, x = grid_location

            if action_label == "UP":
                next_y = y - 1 if y > 0 else y
                next_x = x
            elif action_label == "DOWN":
                next_y = y + 1 if y < (grid_dims[0]-1) else y
                next_x = x
            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x
                next_y = y
            elif action_label == "RIGHT":
                next_x = x + 1 if x < (grid_dims[1]-1) else x
                next_y = y
            elif action_label == "STAY":
                next_x = x
                next_y = y

            new_location = (next_y, next_x)
            next_state = loc_list.index(new_location)
            B[0][next_state, curr_state, action_id] = 1.0

    B[1][:,:,0] = np.eye(num_states[1])
    B[2][:,:,0] = np.eye(num_states[2])

    # The prior preferences: Vector C
    C = utils.obj_array_zeros(num_obs)

    C[3][1] = 2.0 # make the agent want to encounter the "Cheese" observation level
    C[3][2] = -4.0 # make the agent not want to encounter the "Shock" observation level

    # Prior over (initial) hidden states: the D vectors
    D = utils.obj_array_uniform(num_states)
    D[0] = utils.onehot(loc_list.index((0,0)), num_grid_points)

    agent = Agent(A = A, B = B, C = C, D = D, policy_len = 4)

    loc_obs = info['loc_obs']
    cue_1_obs = info['cue_1_obs']
    cue_2_obs = info['cue_2_obs']
    reward_obs = info['reward_obs']

    history_of_locs = [loc_obs]

    obs = [loc_list.index(loc_obs), cue_1_names.index(cue_1_obs), cue_2_names.index(cue_2_obs), reward_names.index(reward_obs)]
    
    while current_iteration <= iteration_limitation:
        print("Experiment #", current_iteration)
        current_step = 0
        if reset:
            reset = False

        while current_step < step_limitation and not reset:
    
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    done = True

            if done:
                break
            
            qs = agent.infer_states(obs)

            agent.infer_policies()

            chosen_action_id = agent.sample_action()

            movement_id = int(chosen_action_id[0])

            choice_action = actions[movement_id]

            observation, reward_obs, done, info = env.step({
                'next_action': choice_action,
                'movement_id': movement_id,
                'chosen_action_id': chosen_action_id,
                'current_step': current_step,
                'current_iteration': current_iteration})
            
            current_step = info['current_step']
            current_iteration = info['current_iteration']

            print(f'Action at step {current_step}: {choice_action}')

            reset = info['reset']

            loc_obs = info['loc_obs']
            cue_1_obs = info['cue_1_obs']
            cue_2_obs = info['cue_2_obs']
            reward_obs = info['reward_obs']

            obs = [loc_list.index(loc_obs), cue_1_names.index(cue_1_obs), cue_2_names.index(cue_2_obs), reward_names.index(reward_obs)]

            history_of_locs.append(loc_obs)

            print(f'Grid location at step {current_step}: {loc_obs}')

            print(f'Cue 2 Obs at step {current_step}: {cue_2_obs} at index {cue_2_names.index(cue_2_obs)}')

            print(f'Reward at step {current_step}: {reward_obs}')
            
            env.render()
            # Control the frame rate (limit to 1 frames per second)
            env.clock.tick(32)
    
    env.close()

if __name__ == "__main__":
    run_environment()
