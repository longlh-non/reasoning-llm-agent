from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

# Define a simple reasoning tool
class ReasoningTool(BaseTool):
    name = "Reasoning"
    description = "Performs reasoning process to infer the next action."

    def _run(self, query: str) -> str:
        # Simulate some reasoning based on the query
        try:
            # Here, the reasoning could be as simple or complex as necessary
            # This is just a placeholder for the actual reasoning logic
            return f"Based on the query: {query}, let's take action X."
        except Exception as e:
            return str(e)

    async def _arun(self, query: str) -> str:
        # For async version (if needed)
        pass


class LLMAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)

        self.instructions = """
You are an agent navigating a grid world of dimension {grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.

One location in the grid world contains a cue: CUE 1. There will be four additional locations that will serve as possible locations for a second cue: CUE 2. Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4. Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

These two possible reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
- [TOP, BOTTOM]: "TOP" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "BOTTOM" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
- [LEFT, RIGHT]: "LEFT" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "RIGHT" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

These are the actions you can only do: UP, DOWN, LEFT, RIGHT, STAY and you can only perform and move one location per state.

If you reach CUE 1, ask for CUE 2 information.

If you reach CUE 2, ask for REWARD information.

ENVIRONMENT SET UP: {environment_setup}

Reasoning: {agent_scratchpad}
Available tools: {tool_names}
You have access to the following tools:
{tools}

Your current location is {current_location}.
Based on the above information, decide the next action to take.

Return only the JSON object with no additional text or formatting:
{{
    \"location\": \"the current location in the format (y, x)\",
    \"action\": \"you should inference for the next action. Then, tell the user about the next action - LEFT, RIGHT, UP, DOWN, STAY))\",
    \"next_location\": \"your next location in the format (y, x)\",
    \"current_location_name\": \"your current location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\",
    \"next_location_name\": \"your next location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\"
}}

If you are at the CUE location, the next action should be STAY and wait for new information.
"""
        self.action_parser = RegexParser(
            regex=r"Action: (.*)", output_keys=["action"], default_output_key="action"
        )

        self.message_history = []
        self.ret = 0

    def random_action(self):
        action = self.env.action_space.sample()
        return action

    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
            SystemMessage(content=self.instructions),
        ]

    def observe(self, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew

        obs_message = f"""
            Observation: {obs}
            Reward: {rew}
            Termination: {term}
            Truncation: {trunc}
            Return: {self.ret}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        return obs_message