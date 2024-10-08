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
from LLMActiveInferenceAgent import LLMActiveInferenceAgent
from utils import parse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

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

    env = gym.make('POMDPGridWorldEnv-v0', start_pos = (5, 5), 
                   is_random_start = False,
                   is_random_reward = False,
                   step_limitation = step_limitation,
                   iteration_limitation = iteration_limitation,
                   type = 'seen')
    
    observation, info = env.reset()
    env.render()  

    done = False

    agent = LLMAgent(llm, env)
    agent.reset()

    while current_iteration < iteration_limitation:
        print("Experiment #", current_iteration)
        current_step = 0
        
        if reset:
                # env.reset()
                agent.reset()
                reset = False
        
        while current_step <= step_limitation and not reset:

            # reset = False
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
            # print(f"Observation: {observation}, Action: {agent_response['action']}, Reward: {reward_obs}, Done: {done}")

            # Control the frame rate (limit to 1 frames per second)
            env.clock.tick(32)
    
    env.close()

if __name__ == "__main__":
    run_environment()
