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



# Test environment
def run_environment():
        
    # Register environment
    register(
        id='POMDPGridWorldEnv-v0',
        entry_point='pomdp-grid-world-env:POMDPGridWorldEnv',
    )

    # LangChain setup
    # Set up OpenAI API key - MAKE THIS KEY PRIVATE
    os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual API key

    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", model_kwargs={ "response_format": { "type": "json_object" }})
    current_iteration = 0
    step_limitation = 20
    iteration_limitation = 50
    reset = False

    # Create the environment
    env = gym.make('POMDPGridWorldEnv-v0',
                   is_random_grid = True,
                   is_random_start = True,
                   is_random_reward = True,
                   is_random_cue_1 = True,
                   is_random_cue_2_locs = True,
                   step_limitation = step_limitation,
                   iteration_limitation = iteration_limitation)
    
    observation, info = env.reset()
    env.render()  

    done = False

    agent = LLMAgent(llm, env)
    agent.reset()

    print ('iteration_times < maximum_iterations: ', current_iteration < iteration_limitation)
    while current_iteration < iteration_limitation:
        print("Experiment #", current_iteration)
        current_step = 0
        if reset:
                # observation, info = env.reset()
                # current_iteration = info['current_iteration']
                agent.reset()
                reset = False

        while current_step < step_limitation and not reset:
           

            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    done = True

            if done:
                break
            agent_response = agent.act()

            observation, reward_obs, done, info = env.step({
                'next_action': agent_response['next_action'],
                'action_reason': agent_response['action_reason'],
                'position': agent_response['position'],
                'next_position': agent_response['next_position'],
                'current_step': current_step,
                'current_iteration': current_iteration})
            
            current_step = info['current_step']
            current_iteration = info['current_iteration']
            print("Step #", current_step)

            reset = info['reset']
            agent_response['reset'] = info['reset']
            
            obs_message = agent.observe(llm_obs=agent_response, obs = observation)

            print('agent_response: ', agent_response)
            print(f"Next action: {agent_response['next_action']}")
            
            env.render()
            # Control the frame rate (limit to 1 frames per second)
            env.clock.tick(32)
    
    env.close()

if __name__ == "__main__":
    run_environment()
