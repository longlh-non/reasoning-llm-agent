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


reasoning_tool = Tool(
            name="Reasoning",
            func=ReasoningTool().run,
            description="Helps in reasoning about the next move in the grid world."
        )

# Define the tool that the agent can use
# tools = [reasoning_tool]

# llm_with_tools = llm.bind_functions([reasoning_tool, ModelResponse])

# Create the ReAct agent using the defined prompt and tools
# agent = create_react_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt,
#     output_parser=parse)


# Initialize the agent executor
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Test environment
def run_environment():
        
    # Register environment
    register(
        id='POMDPGridWorldEnv-v0',
        entry_point='pomdp-grid-world-env:POMDPGridWorldEnv',
    )

    # LangChain setup
    # Set up OpenAI API key
    os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual API key

    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o", model_kwargs={ "response_format": { "type": "json_object" }})

    # Create the environment
    env = gym.make('POMDPGridWorldEnv-v0')
    observation, info = env.reset()
    env.render()  

    done = False

    agent = LLMAgent(llm, env)
    agent.reset()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            else:         
                agent_response = agent.act()
                observation, reward_obs, done, info = env.step(agent_response['action'])
                obs_message = agent.observe(llm_obs=agent_response, obs = observation)
                print('agent_response: ', agent_response)
                print(f"Action: {agent_response['action']}")
                print('obs_message: ', obs_message)
                
                env.render()
                print(f"Observation: {observation}, Action: {agent_response['action']}, Reward: {reward_obs}, Done: {done}")

    env.close()

if __name__ == "__main__":
    run_environment()
