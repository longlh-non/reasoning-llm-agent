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

# Define the prompt template for the agent
# prompt_template = """
# You are an agent navigating a grid world of dimension {grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.

# One location in the grid world contains a cue: CUE 1. There will be four additional locations that will serve as possible locations for a second cue: CUE 2. Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4. Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

# These two possible reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
# - [TOP, BOTTOM]: "TOP" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "BOTTOM" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
# - [LEFT, RIGHT]: "LEFT" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "RIGHT" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

# These are the actions you can only do: UP, DOWN, LEFT, RIGHT, STAY and you can only perform and move one location per state.

# If you reach CUE 1, ask for CUE 2 information.

# If you reach CUE 2, ask for REWARD information.

# ENVIRONMENT SET UP: {environment_setup}

# Reasoning: {agent_scratchpad}
# Available tools: {tool_names}
# You have access to the following tools:
# {tools}

# Your current location is {current_location}.
# Based on the above information, decide the next action to take.

# Return only the JSON object with no additional text or formatting:
# {{
#     \"location\": \"the current location in the format (y, x)\",
#     \"action\": \"you should inference for the next action. Then, tell the user about the next action - LEFT, RIGHT, UP, DOWN, STAY))\",
#     \"next_location\": \"your next location in the format (y, x)\",
#     \"current_location_name\": \"your current location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\",
#     \"next_location_name\": \"your next location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\"
# }}

# If you are at the CUE location, the next action should be STAY and wait for new information.
# """

reasoning_tool = Tool(
            name="Reasoning",
            func=ReasoningTool().run,
            description="Helps in reasoning about the next move in the grid world."
        )

# prompt = PromptTemplate(
#     input_variables=["grid_world_dimension", "environment_setup", "current_location", "tool_names", "tools", "agent_scratchpad"],
#     template=prompt_template
# )

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

    # Convert the grid info to a string format that the agent can process
    # grid_info_str = f"""
    # grid_world_dimension: {env.grid_world_dimension},
    # start: {env.start},
    # cue_1_location: {env.cue_1_location},

    # """
    # cue_2_locations: {{
    #     L1: {grid_info['cue_2_locations']['L1']},
    #     L2: {grid_info['cue_2_locations']['L2']},
    #     L3: {grid_info['cue_2_locations']['L3']},
    #     L4: {grid_info['cue_2_locations']['L4']}
    # }},
    # reward_locations: {{
    #     LEFT: {grid_info['reward_locations']['LEFT']},
    #     RIGHT: {grid_info['reward_locations']['RIGHT']}
    # }}
    # Define the current location of the agent

    # Format tools into a string to include in the prompt
    # formatted_tools = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])


    done = False

    agent = LLMAgent(llm, env)
    agent.reset()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = event.pos
                    if env.grid_size // 2 - 50 <= mouse_pos[0] <= env.grid_size // 2 + 50 and env.grid_size + 10 <= mouse_pos[1] <= env.grid_size + 40:
                        # action = env.action_space.sample()  # Random action
                        
                        # Run the agent with the current setup
                        # agent_response = agent_executor.invoke({
                        #         "grid_world_dimension": env.grid_world_dimension,
                        #         "environment_setup": grid_info_str,
                        #         "current_location": env.agent_pos,
                        #         "tool_names": "Reasoning",
                        #         "tools": formatted_tools,
                        #         "agent_scratchpad": ""
                        #     },     
                        #     return_only_outputs=True,)
                        
                        agent_response = agent.act()
                        # agent_response_json = json.loads(agent_response)
                        observation, reward_obs, done, info = env.step(agent_response['action'])
                        obs_message = agent.observe(llm_obs=agent_response, obs = observation)
                        print('agent_response: ', agent_response)
                        print(f"Action: {agent_response['action']}")
                        print(obs_message)
                        
                        env.render()
                        print(f"Observation: {observation}, Action: {agent_response['action']}, Reward: {reward_obs}, Done: {done}")

    env.close()

if __name__ == "__main__":
    run_environment()
