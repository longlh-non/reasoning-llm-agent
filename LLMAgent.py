from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import json

class LLMAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    # GIVE THE LLM AN EXAMPLE
    # THINK ABOUT TREE OF THOUGHT
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)
        self.instructions = f"""
            You are an agent navigating a grid world of dimension {env.grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.
            
            You should visualize the grid world as a matrix.
            
            One location in the grid world contains a cue: CUE 1. 
            There will be four additional locations that will serve as possible locations for a second cue: CUE 2. 
            Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4.
            Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

            These two possible types of REWARD CONDITIONS which are reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
            - [TOP, BOTTOM]: "TOP" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "BOTTOM" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
            - [LEFT, RIGHT]: "LEFT" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "RIGHT" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

            These are the actions you can only do: MOVE_UP meaning you will move upward one cell, MOVE_DOWN meaning you will move upward one cell, MOVE_LEFT meaning you will move left one cell, MOVE_RIGHT meaning you will move right one cell, STAY meaning you stay at current cell and you can only perform and move one cell per state.

            If you reach CUE 1, the next action should be STAY and CUE 2 signals will be revealed from 4 given ones which are L1, L2, L3, L4 by Human message.

            Afterward, the next goal is to reach CUE 2 location.
            
            If you reach CUE 2, the next action should be STAY and the REWARD CONDITION will be revealed by Human message.

            Afterward, the next goal is to reach CHEESE's location meanwhile avoid SHOCK's location.
            
            ENVIRONMENT SET UP: {env.environment_setup}

            Your current location is {env.agent_pos}.

            Based on the above information, decide the next action to take.

            Return only the JSON object with no additional text or formatting:
            {{
                \"position\": \"(y, x) which is current agent position\",
                \"next_action\": \"You should inference for the next action. Remember to compare your current location and the location of cue_1, cue_2, cheese and shock given by Human. Then, tell the user about the next action - MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY))\",
                \"action_reason\": \"explain why you perform above ation and also compare the location of cue_1, cue_2, cheese and shock given by Human to verify the explaination you gave. If you are going to reveal a location, please give the reason why.\",
                \"next_position\": \"(y, x) which is next agent position after reasoning and performing the next_action \",
                \"current_goal\": \"the name of agent's current goal such as  cue_1, L1, L2, L3, L4 or cheese\",
                \"cheese_location\": \"the location of cheese in format (y, x) or Null if you don't know where it is\",
                \"shock_location\": \"the location of shock in format (y, x) or Null if you don't know where it is\",
            }}

            FOLLOW THIS INSTRUCTION BELOW AS AN SAMPLE ONLY, YOU HAVE TO USE THE INFORMATION COMES FROM ENVIRONMENT SETUP AND HUMAN MESSAGE FOR REASONING PROCESS.
            THE INFORMATION GIVEN BELOW IS JUST A MOCK EXPERIMENT, YOU MUST NOT USE THE INFORMATION GIVEN IN THE FOLLOWING QUOTES FOR REASONING PROCESS:
                \"The Grid world Dimension is: {self.env.grid_world_dimension}
                    Current location of agent is (1, 4),
                    You have to inference to move to the location of CUE 1 at (2, 1) (SAMPLE ONLY, NOT THE REAL CUE 1 LOCATION). 
                    Then there are four additional locations that will serve as possible locations for CUE 2 (SAMPLE ONLY, NOT THE REAL CUE 2 LOCATIONS) which are {{\"L1"\: (1, 3), \"L2"\: (2, 4), \"L3"\: (4, 4), \"L4"\: (5, 3)}}.
                    The one is revealed are L3 and need to reach it.
                    After reaching it, there are new informations which are reward conditions named which are {{\"LEFT"\: (2, 0), \"RIGHT"\: (2, 3)}} and the one is revealed as CHEESE is LEFT so that RIGHT is SHOCK.
                    You need to infering and reach CHEESE on (2, 0) (SAMPLE ONLY, NOT THE REAL CHEESE LOCATION) while avoiding SHOCK on (2, 3) (SAMPLE ONLY, NOT THE REAL SHOCK LOCATION).

                    The SAMPLE output should be: {{
                        \"position\": \"(1, 4)\",
                        \"next_action\": \"MOVE_DOWN\",
                        \"action_reason\": \"Because cue_1 is on (2, 1), perform MOVE_DOWN to move downward one cell to have the same horizontal axe with cue_1 (2, 4)\",
                        \"next_position\": \"(2, 4)\",
                        \"current_goal\": \"cue_1\",                        
                        \"cheese_location\": \"Null\"
                        \"shock_location\": \"Null\",
                    }}
                \"
            
            TRY TO GENERATE LEAST TOKENS AS YOU CAN TOO IMPROVE PERFORMANCE BUT STILL KEEP THE SAME REASONING METHOD FOR YOUR ACTION.
        """

        self.message_history = []
        self.ret = 0
        self.log_file = "chat_history.txt"

    def random_action(self):
        action = self.env.action_space.sample()
        return action

    def reset(self):
        self.reset_instruction()
        self.message_history = [
            SystemMessage(content=self.instructions),
        ]
        self.reset_log_file()
        self.log_conversation(self.instructions)

    def reset_instruction(self):
        self.instructions = f"""
            Let's think step-by-step. 

            You are an agent navigating a grid world of dimension {self.env.grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.
            
            You should visualize the grid world as a matrix.
            
            One location in the grid world contains a cue: CUE 1. 
            There will be four additional locations that will serve as possible locations for a second cue: CUE 2. 
            Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4.
            Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

            These two possible types of REWARD CONDITIONS which are reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
            - [TOP, BOTTOM]: "TOP" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "BOTTOM" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
            - [LEFT, RIGHT]: "LEFT" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "RIGHT" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

            These are the actions you can only do: MOVE_UP meaning you will move upward one cell, MOVE_DOWN meaning you will move upward one cell, MOVE_LEFT meaning you will move left one cell, MOVE_RIGHT meaning you will move right one cell, STAY meaning you stay at current cell and you can only perform and move one cell per state.

            If you reach CUE 1, the next action should be STAY and CUE 2 signals will be revealed from 4 given ones which are L1, L2, L3, L4 by Human message.

            Afterward, the next goal is to reach CUE 2 location.
            
            If you reach CUE 2, the next action should be STAY and the REWARD CONDITION will be revealed by Human message.

            Afterward, the next goal is to reach CHEESE's location meanwhile avoid SHOCK's location.
            
            ENVIRONMENT SET UP: {self.env.environment_setup}

            Your current location is {self.env.agent_pos}.

            Based on the above information, decide the next action to take.

            Return only the JSON object with no additional text or formatting:
            {{
                \"position\": \"(y, x) which is current agent position\",
                \"next_action\": \"You should inference for the next action. Remember to compare your current location and the location of cue_1, cue_2, cheese and shock given by Human. Then, tell the user about the next action - MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY))\",
                \"action_reason\": \"explain why you perform above ation and also compare the location of cue_1, cue_2, cheese and shock given by Human to verify the explaination you gave. If you are going to reveal a location, please give the reason why.\",
                \"next_position\": \"(y, x) which is next agent position after reasoning and performing the next_action \",
                \"current_goal\": \"the name of agent's current goal such as  cue_1, L1, L2, L3, L4 or cheese\",
                \"cheese_location\": \"the location of cheese in format (y, x) or Null if you don't know where it is\",
                \"shock_location\": \"the location of shock in format (y, x) or Null if you don't know where it is\",
            }}

            FOLLOW THIS INSTRUCTION BELOW AS AN SAMPLE ONLY, YOU HAVE TO USE THE INFORMATION COMES FROM ENVIRONMENT SETUP AND HUMAN MESSAGE FOR REASONING PROCESS.
            THE INFORMATION GIVEN BELOW IS JUST A MOCK EXPERIMENT, YOU MUST NOT USE THE INFORMATION GIVEN IN THE FOLLOWING QUOTES FOR REASONING PROCESS:
                \"The Grid world Dimension is: {self.env.grid_world_dimension}
                    Current location of agent is (1, 4),
                    You have to inference to move to the location of CUE 1 at (2, 1) (SAMPLE ONLY, NOT THE REAL CUE 1 LOCATION). 
                    Then there are four additional locations that will serve as possible locations for CUE 2 (SAMPLE ONLY, NOT THE REAL CUE 2 LOCATIONS) which are {{\"L1"\: (1, 3), \"L2"\: (2, 4), \"L3"\: (4, 4), \"L4"\: (5, 3)}}.
                    The one is revealed are L3 and need to reach it.
                    After reaching it, there are new informations which are reward conditions named which are {{\"LEFT"\: (2, 0), \"RIGHT"\: (2, 3)}} and the one is revealed as CHEESE is LEFT so that RIGHT is SHOCK.
                    You need to infering and reach CHEESE on (2, 0) (SAMPLE ONLY, NOT THE REAL CHEESE LOCATION) while avoiding SHOCK on (2, 3) (SAMPLE ONLY, NOT THE REAL SHOCK LOCATION).

                    The SAMPLE output should be: {{
                        \"position\": \"(1, 4)\",
                        \"next_action\": \"MOVE_DOWN\",
                        \"action_reason\": \"Because cue_1 is on (2, 1), perform MOVE_DOWN to move downward one cell to have the same horizontal axe with cue_1 (2, 4)\",
                        \"next_position\": \"(2, 4)\",
                        \"current_goal\": \"cue_1\",
                        \"cheese_location\": \"Null\"
                        \"shock_location\": \"Null\",
                    }}
                \"
            TRY TO GENERATE AS LEAST TOKENS AS YOU CAN TO IMPROVE SPEED.
        """

    def reset_log_file(self):
        with open(self.log_file, 'w') as file:
            file.write("Conversation Log\n")
    
    def log_info(self, info):
        with open('debug_agent_info.txt', 'a') as file:
            file.write(f"{info}\n")

    def log_conversation(self, obs_msg):
        with open(self.log_file, 'a') as file:
            log_entry = f"{obs_msg}\n"
            file.write(log_entry)

    def observe(self, llm_obs, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew
        obs_message = ''
        # obs_message = f"""
        #     Observation: {obs}
        #     Action: {obs}
        #     Reward: {rew}
        #     Termination: {term}
        #     Truncation: {trunc}
        #     Return: {self.ret}
        # """
        if llm_obs['reset'] == True:
            self.reset()
        else:
            obs_message = f'The agent is now at {llm_obs['position']} and about to {llm_obs['next_action']} to {llm_obs['next_position']} in order to find {llm_obs['current_goal']}. Please move to {llm_obs['next_position']} and infer for the next step. If you are trying to reach cheese after reaching cue 2 and shock is on the way. Find another path.'

            if llm_obs['position'] == str(self.env.cue_1_location):
                self.message_history.append(SystemMessage(content='WHAT IS CUE 2 NAME?'))
                obs_message = f"These are cue 2 possible locations: {{\"L1\": {self.env.cue_2_locations[0]}, \"L2\": {self.env.cue_2_locations[1]}, \"L3\": {self.env.cue_2_locations[2]}, \"L4\": {self.env.cue_2_locations[3]}}} and the one specified as cue_2 is {self.env.cue_1_obs}, the other locations are empty now. Keep Infering until reaching {self.env.cue_1_obs}"

            if llm_obs['position'] == str(self.env.cue_2_location):
                self.message_history.append(SystemMessage(content='WHAT IS REWARD CONDITION?'))
                obs_message = f"These are possible reward locations: {{\"{self.env.reward_conditions[0]}\": {self.env.reward_locations[0]}, \"{self.env.reward_conditions[1]}\": {self.env.reward_locations[1]}}} and the CHEESE is {self.env.cue_2_obs} so that the SHOCK is the other one in possible reward locations is SHOCK. Keep Infering until reaching the CHEESE and try to avoid the SHOCK"
            self.message_history.append(HumanMessage(content=obs_message))
            self.log_conversation(obs_message)
            
            # CHECK IF LLM CAN UNDERSTAND OR NOT
            if llm_obs['position'] == str(self.env.prev_reward_location):
                if self.env.reward_obs == 'SHOCK':    
                    self.message_history.append(SystemMessage(content='EXPERIMENT FAILED'))
                else:
                    self.message_history.append(SystemMessage(content='EXPERIMENT SUCCESS'))
                obs_message = f"Let try every step again with a new starting location: {self.env.start}"
                
                # if self.env.is_random_start == False:
                self.reset()

        return obs_message
    
    def _act(self):
        result_message = self.model.invoke(self.message_history)
        self.log_conversation(result_message)
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
    