from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import json

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
        self.instructions = f"""
            You are an agent navigating a grid world of dimension {env.grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.

            One location in the grid world contains a cue: CUE 1. 
            There will be four additional locations that will serve as possible locations for a second cue: CUE 2. 
            Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4.
            Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

            These two possible types of REWARD CONDITIONS which are reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
            - [TOP, BOTTOM]: "TOP" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "BOTTOM" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
            - [LEFT, RIGHT]: "LEFT" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "RIGHT" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

            These are the actions you can only do: UP, DOWN, LEFT, RIGHT, STAY and you can only perform and move one location per state.

            If you reach CUE 1, ask for CUE 2 information.

            If you reach CUE 2, ask for REWARD information.

            ENVIRONMENT SET UP: {env.environment_setup}

            Your current location is {env.agent_pos}.

            Based on the above information, decide the next action to take.

            Return only the JSON object with no additional text or formatting:
            {{
                \"location\": \"(y, x) which is current agent position\",
                \"action\": \"you should inference for the next action. Then, tell the user about the next action - LEFT, RIGHT, UP, DOWN, STAY))\",
                \"next_location\": \"(y, x) which is next agent position\",
                \"current_location_name\": \"your current location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\",
                \"next_location_name\": \"your next location in the format (y, x) or cue_1 (if you are ons on CUE 1) or cue_2 (CUE 2 location) or cheese (if you are on cheese) or shock (shock location)\",
                \"current_goal_location\": \"your current goal location in the format (y, x)\"
            }}

            Example: {{
                \"location\": \"(2, 0)\",
                \"action\": \"RIGHT\",
                \"next_location\": \"(2, 1)\",
                \"current_location_name\": \"(2, 1) or cue_1 or cue_2 or cheese or shock\",
                \"next_location_name\": \"(2, 2) or cue_1 or cue_2 or cheese or shock\",
                \"current_goal_location\": \"(2, 2)\"
            }}

            If you are at the CUE location, the next action should be STAY and wait for new observation to perform an ACTION.
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
            SystemMessage(content=self.instructions),
        ]

    def observe(self, llm_obs, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew

        # obs_message = f"""
        #     Observation: {obs}
        #     Action: {obs}
        #     Reward: {rew}
        #     Termination: {term}
        #     Truncation: {trunc}
        #     Return: {self.ret}
        # """
        obs_message = 'KEEP INFERING'

        if llm_obs['location'] == str(self.env.cue_1_location):
            self.message_history.append(SystemMessage(content='WHAT IS CUE 2 NAME?'))
            obs_message = f"These are cue_2_locations: {{\"L1\": {self.env.cue_2_locations[0]}, \"L2\": {self.env.cue_2_locations[1]}, \"L3\": {self.env.cue_2_locations[2]}, \"L4\": {self.env.cue_2_locations[3]}}} and cue_2 is {self.env.cue_1_obs}. Keep Infering until reaching CUE 2"

        print("self.env.cue_2_location: ", type(str(self.env.cue_2_location)))
        print("self.env.cue_2_location: ", str(self.env.cue_2_location))
        print("llm_obs['location']: ", type(llm_obs['location']))
        print("llm_obs['location']: ", llm_obs['location'])

        print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE: ", llm_obs['location'] == str(self.env.cue_2_location))

        
        if llm_obs['location'] == str(self.env.cue_2_location):
            self.message_history.append(SystemMessage(content='WHAT IS REWARD CONDITION?'))
            obs_message = f"These are reward_conditions: {self.env.reward_conditions} which is located at {self.env.reward_locations} and reward condition is {self.env.reward_condition}. Keep Infering until reaching the reward condition"

        print('obs_message: ', obs_message)

        self.message_history.append(HumanMessage(content=obs_message))

        return obs_message
    
    def _act(self):
        result_message = self.model.invoke(self.message_history)
        self.message_history.append(result_message.content)
        # action = int(self.action_parser.parse(act_message.content)["action"])
        agent_response = json.loads(result_message.content)
        return agent_response

    def act(self):
        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_none(),  # No waiting time between retries
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: print(
                    f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
                ),
            ):
                with attempt:
                    action = self._act()
        except tenacity.RetryError:
            action = self.random_action()
        return action
    