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
from LLMAgent import ReasoningTool, LLMAgent
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


    # CHECK IF NEED A NEW AGENT - CURRENT RUNIING THE SAME AGENT EVERY TIME WITH THE SAME INSTANCE
    # TRY NEW INSTANCE - SAME INSTANCE
    # TRY NEW AGENT
    # TRY NEW ENV
    # TRY SAME EVERYTHING

    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", model_kwargs={ "response_format": { "type": "json_object" }})

    # Create the environment
    env = gym.make('POMDPGridWorldEnv-v0')
    observation, info = env.reset()
    env.render()  

    done = False

    agent = LLMAgent(llm, env)
    agent.reset()
    iteration_times = 0
    maximum_iteration_times = 50
    while iteration_times < maximum_iteration_times:
        iteration_times+=1
        print("Experiments #", iteration_times)
        # ADD INFERING TIMES LIMITATION
        infering_times = 0
        while not done:
            for event in pygame.event.get():
                infering_times+=1
                if infering_times > 25:
                    infering_times = 0

                if event.type == pygame.QUIT:
                    done = True
                else:         
                    agent_response = agent.act()
                    observation, reward_obs, done, info = env.step({'action': agent_response['action'], 'infering_times': infering_times})
                    agent_response['reset'] = info['reset']
                    obs_message = agent.observe(llm_obs=agent_response, obs = observation)
                    print("infering_times: ", infering_times)
                    print(f"event.type at infering_time {infering_times}: {event.type}")
                    print('agent_response: ', agent_response)
                    print(f"Action: {agent_response['action']}")
                    
                    env.render()
                    # print(f"Observation: {observation}, Action: {agent_response['action']}, Reward: {reward_obs}, Done: {done}")

        env.close()

if __name__ == "__main__":
    run_environment()
