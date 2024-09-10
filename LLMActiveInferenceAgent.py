from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import json
import numpy as np

class LLMActiveInferenceAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env):
        self.model = model
        self.env = env

        # Define Environment and Agent Parameters
        num_grid_points = np.prod([env.row, env.column]) # total number of grid locations (rows X columns)
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
        self.cue_1_location = env.cue_1_location
        self.cue_2_location = env.cue_2_location
        self.reward_locations = env.reward_locations

        # Initialize Observation Matrices: A
        self.num_states = [num_grid_points, len(env.cue_2_locations), len(env.reward_conditions)]
        self.cue_1_names = ['Null'] + env.cue_2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
        self.cue_2_names = ['Null'] + env.reward_conditions
        self.reward_names = ['Null', 'Cheese', 'Shock']
        self.num_obs = [num_grid_points, len(self.cue_1_names), len(self.cue_2_names), len(self.reward_names)]

        # Observation array model: A array
        self.A_m_shapes = [ [o_dim] + self.num_states for o_dim in self.num_obs] # list of shapes of modality-specific A[m] arrays
        self.A = np.empty(len(self.A_m_shapes), dtype=object) # initialize A array to an object array of all-zero subarrays
        
        # Fill A with all-zero subarrays of the shapes specified in A_m_shapes
        for i, shape in enumerate(self.A_m_shapes):
            self.A[i] = np.zeros(shape)

        # make the location observation only depend on the location state (proprioceptive observation modality)
        self.A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, self.num_states[1], self.num_states[2]))

        # make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
        self.A[1][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

        # Make the Cue 1 signal depend on 1) being at the Cue 1 location an) the location of Cue 2
        for i, cue_loc_2_i in enumerate(env.cue_2_locations):
            self.A[1][0,loc_list.index(self.cue_1_location),i,:] = 0.0
            self.A[1][i+1,loc_list.index(self.cue_1_location),i,:] = 1.0

        # make the reward observation depend on the location (being at reward location) and the reward condition
        self.A[3][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

        rew_1st_idx = loc_list.index(self.reward_locations[0]) # linear index of the location of the "FIRST" reward location
        rew_2nd_idx = loc_list.index(self.reward_locations[1]) # linear index of the location of the "SECOND" reward location

        # fill out the contingencies when the agent is in the "FIRST" reward location
        self.A[3][0,rew_1st_idx,:,:] = 0.0
        self.A[3][1,rew_1st_idx,:,0] = 1.0
        self.A[3][2,rew_1st_idx,:,1] = 1.0

        # fill out the contingencies when the agent is in the "SECOND" reward location
        self.A[3][0,rew_2nd_idx,:,:] = 0.0
        self.A[3][1,rew_2nd_idx,:,1] = 1.0
        self.A[3][2,rew_2nd_idx,:,0] = 1.0

        # make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
        self.A[2][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

        for i, cue_loc2_i in enumerate(env.cue_2_locations):

            # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
            self.A[2][0,loc_list.index(cue_loc2_i),i,:] = 0.0
            self.A[2][1,loc_list.index(cue_loc2_i),i,0] = 1.0
            self.A[2][2,loc_list.index(cue_loc2_i),i,1] = 1.0

        self.docs = self.get_docs(env)

        # Cue 1 location in 1D array
        self.cue_1_1d_idx = self.env.cue_1_location[1]*self.env.column+self.env.cue_1_location[0]
        
        self.grid_size = env.row*env.column
        self.instructions = f"""
            You are an agent navigating a grid world of dimension {env.grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.
            
            You should visualize the grid world as a matrix.
            
            One location in the grid world contains a cue: CUE 1. 
            There will be four additional locations that will serve as possible locations for a second cue: CUE 2. 
            Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4.
            Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

            These two possible types of REWARD CONDITIONS which are reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
            - [FIRST, SECOND]: "FIRST" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "SECOND" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
            - [FIRST, SECOND]: "FIRST" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "SECOND" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

            These are the actions you can only do: MOVE_UP meaning you will move upward one cell, MOVE_DOWN meaning you will move upward one cell, MOVE_LEFT meaning you will move left one cell, MOVE_RIGHT meaning you will move right one cell, STAY meaning you stay at current cell and you can only perform and move one cell per state.

            If you reach CUE 1, the next action should be STAY and CUE 2 signals will be revealed from 4 given ones which are L1, L2, L3, L4 by Human message.

            Afterward, the next goal is to reach CUE 2 location.
            
            If you reach CUE 2, the next action should be STAY and the REWARD CONDITION will be revealed by Human message.

            Afterward, the next goal is to reach CHEESE's location meanwhile avoid SHOCK's location.
            
            ENVIRONMENT SET UP: {env.environment_setup}

            Your current location is {env.agent_pos}.

            Based on the above information, decide the next action to take.

            ### Environment Setup:
            1. The grid world has {env.row*env.column} possible locations for the agent.
            2. There are 4 possible locations for Cue 2: {env.cue_2_locations}.
            3. There are 2 reward conditions: 'FIRST' (reward at the first location) or 'SECOND' (reward at the second location).

            ### Observation Matrix Structure:
            The observation matrix should represent the probability of receiving certain observations given the agent's hidden states:
            1. `A[0]`: Location observation matrix, an identity matrix of size [{self.grid_size} x {self.grid_size} x {len(env.cue_2_locations)} x {len(env.reward_locations)}].
            2. `A[1]`: Cue 1 observation matrix, size [{len(self.cue_1_names)} x {self.grid_size} x {len(env.cue_2_locations)} x {len(env.reward_locations)}], indicating where Cue 2 is located.
            3. `A[2]`: Cue 2 observation matrix, size [{len(self.cue_2_names)} x {self.grid_size} x {len(env.cue_2_locations)} x {len(env.reward_locations)}], indicating the reward condition.
            4. `A[3]`: Reward observation matrix, size [{len(self.reward_names)} x {self.grid_size} x {len(env.cue_2_locations)} x {len(env.reward_locations)}], indicating whether the agent receives 'Cheese', 'Shock', or 'Null'.

            ### Rules for Filling the Matrix:
            1. `A[0]` should be an identity matrix, where A[0][i, i, :, :] = 1 for all i from 0 to {self.grid_size-1}.
            2. `A[1]`: 
            - Default to 'Null' everywhere (A[1][0, :, :, :] = 1.0).
            - At Cue 1 location {self.env.cue_1_location}, provide signals for Cue 2's location:
                - A[1][1, {self.cue_1_1d_idx}, 0, :] = 1.0, A[1][2, {self.cue_1_1d_idx}, 1, :] = 1.0, A[1][3, {self.cue_1_1d_idx}, 2, :] = 1.0, A[1][4, {self.cue_1_1d_idx}, 3, :] = 1.0.
            3. `A[2]`: 
            - Default to 'Null' everywhere (A[2][0, :, :, :] = 1.0).
            - At each Cue 2 location, fill according to reward conditions:
                - A[2][1, <Cue 2 index>, i, 0] = 1.0 for 'FIRST'; A[2][2, <Cue 2 index>, i, 1] = 1.0 for 'SECOND'.
            4. `A[3]`: 
            - Default to 'Null' everywhere (A[3][0, :, :, :] = 1.0).
            - At reward locations:
                - A[3][1, first_idx, :, 0] = 1.0 ('Cheese' at 'FIRST'), A[3][2, first_idx, :, 1] = 1.0 ('Shock' at 'FIRST'),
                - A[3][1, second_idx, :, 1] = 1.0 ('Cheese' at 'SECOND'), A[3][2, second_idx, :, 0] = 1.0 ('Shock' at 'SECOND').

            ### Task:
            Please generate the observation matrix (`A` array) according to these guidelines and provide the matrices in Python NumPy array format.
            Return only the JSON object with no additional text or formatting:
            {{
                \"position\": \"(y, x) which is current agent position\",
                \"next_action\": \"You should inference for the next action. Remember to compare your current location and the location of cue_1, cue_2, cheese and shock given by Human. Then, tell the user about the next action - MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY))\",
                \"action_reason\": \"explain why you perform above ation and also compare the location of cue_1, cue_2, cheese and shock given by Human to verify the explaination you gave. If you are going to reveal a location, please give the reason why.\",
                \"next_position\": \"(y, x) which is next agent position after reasoning and performing the next_action \",
                \"current_goal\": \"the name of agent's current goal such as  cue_1, L1, L2, L3, L4 or cheese\",
                \"cheese_location\": \"the location of cheese in format (y, x) or Null if you don't know where it is\",
                \"shock_location\": \"the location of shock in format (y, x) or Null if you don't know where it is\",
                \"obs_matrix\": \"the observation matrix A for active inference agent \",
            }}

            FOLLOW THIS INSTRUCTION BELOW AS AN SAMPLE ONLY, YOU HAVE TO USE THE INFORMATION COMES FROM ENVIRONMENT SETUP AND HUMAN MESSAGE FOR REASONING PROCESS.
            THE INFORMATION GIVEN BELOW IS JUST A MOCK EXPERIMENT, YOU MUST NOT USE THE INFORMATION GIVEN IN THE FOLLOWING QUOTES FOR REASONING PROCESS:
                \"The Grid world Dimension is: {self.env.grid_world_dimension}
                    Current location of agent is (1, 4),
                    You have to inference to move to the location of CUE 1 at (2, 1) (SAMPLE ONLY, NOT THE REAL CUE 1 LOCATION). 
                    Then there are four additional locations that will serve as possible locations for CUE 2 (SAMPLE ONLY, NOT THE REAL CUE 2 LOCATIONS) which are {{\"L1"\: (1, 3), \"L2"\: (2, 4), \"L3"\: (4, 4), \"L4"\: (5, 3)}}.
                    The one is revealed are L3 and need to reach it.
                    After reaching it, there are new informations which are reward conditions named which are {{\"FIRST"\: (2, 0), \"SECOND"\: (2, 3)}} and the one is revealed as CHEESE is FIRST so that SECOND is SHOCK.
                    You need to infering and reach CHEESE on (2, 0) (SAMPLE ONLY, NOT THE REAL CHEESE LOCATION) while avoiding SHOCK on (2, 3) (SAMPLE ONLY, NOT THE REAL SHOCK LOCATION).

                    The SAMPLE output should be: {{
                        \"position\": \"(1, 4)\",
                        \"next_action\": \"MOVE_DOWN\",
                        \"action_reason\": \"Because cue_1 is on (2, 1), perform MOVE_DOWN to move downward one cell to have the same horizontal axe with cue_1 (2, 4)\",
                        \"next_position\": \"(2, 4)\",
                        \"current_goal\": \"cue_1\",                        
                        \"cheese_location\": \"Null\"
                        \"shock_location\": \"Null\",
                        \"obs_matrix\": \"A matrix\",
                    }}
                \"
            You are tasked with constructing an observation matrix for an agent performing active inference in a 5x7 grid world. The agent must navigate this grid to find a hidden reward ('Cheese') while avoiding a punishment ('Shock').
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
            You are an agent navigating a grid world of dimension {self.env.grid_world_dimension} to find a CHEESE and avoiding a SHOCK. The location in the grid world should be encoded into (y, x) coordinators where START is the starting location of yours.
            
            You should visualize the grid world as a matrix.
            
            One location in the grid world contains a cue: CUE 1. 
            There will be four additional locations that will serve as possible locations for a second cue: CUE 2. 
            Crucially, only one of these four additional locations will actually contain CUE 2 - the other 3 will be empty. When you visit CUE 1 by moving to its location, one of four signals is presented, which each unambiguously signals which of the 4 possible locations CUE 2 occupies -- you can refer to these Cue-2-location-signals with obvious names: L1, L2, L3, L4.
            Once CUE 2's location has been revealed, by visiting that location the agent will then receive one of two possible signals that indicate where the hidden reward is located (and conversely, where the hidden punishment lies).

            These two possible types of REWARD CONDITIONS which are reward/punishment locations are indicated by two locations and we have 2 ways to define this: 
            - [FIRST, SECOND]: "FIRST" (meaning the CHEESE reward is on the upper of the two locations and SHOCK punishment is on the lower one) or "SECOND" (meaning the CHEESE reward is on the lower of the two locations and SHOCK punishment is on the upper one).
            - [FIRST, SECOND]: "FIRST" (meaning the CHEESE reward is on the lefter of the two locations and SHOCK punishment is on the righter one) or "SECOND" (meaning the CHEESE reward is on the righter of the two locations and SHOCK punishment is on the lefter one).

            These are the actions you can only do: MOVE_UP meaning you will move upward one cell, MOVE_DOWN meaning you will move upward one cell, MOVE_LEFT meaning you will move left one cell, MOVE_RIGHT meaning you will move right one cell, STAY meaning you stay at current cell and you can only perform and move one cell per state.

            If you reach CUE 1, the next action should be STAY and CUE 2 signals will be revealed from 4 given ones which are L1, L2, L3, L4 by Human message.

            Afterward, the next goal is to reach CUE 2 location.
            
            If you reach CUE 2, the next action should be STAY and the REWARD CONDITION will be revealed by Human message.

            Afterward, the next goal is to reach CHEESE's location meanwhile avoid SHOCK's location.
            
            ENVIRONMENT SET UP: {self.env.environment_setup}

            Your current location is {self.env.agent_pos}.

            Based on the above information, decide the next action to take.

            ### Environment Setup:
            1. The grid world has {self.env.row*self.env.column} possible locations for the agent.
            2. There are 4 possible locations for Cue 2: {self.env.cue_2_locations}.
            3. There are 2 reward conditions: 'FIRST' (reward at the first location) or 'SECOND' (reward at the second location).

            ### Observation Matrix Structure:
            The observation matrix A should represent the probability of receiving certain observations given the agent's hidden states:
            1. `A[0]`: Location observation matrix, an identity matrix of size [{self.env.row*self.env.column} x {self.env.row*self.env.column} x {len(self.env.cue_2_locations)} x {len(self.env.reward_locations)}].
            2. `A[1]`: Cue 1 observation matrix, size [{len(self.cue_1_names)} x {self.env.row*self.env.column} x {len(self.env.cue_2_locations)} x {len(self.env.reward_locations)}], indicating where Cue 2 is located.
            3. `A[2]`: Cue 2 observation matrix, size [{len(self.cue_2_names)} x {self.env.row*self.env.column} x {len(self.env.cue_2_locations)} x {len(self.env.reward_locations)}], indicating the reward condition.
            4. `A[3]`: Reward observation matrix, size [{len(self.reward_names)} x {self.env.row*self.env.column} x {len(self.env.cue_2_locations)} x {len(self.env.reward_locations)}], indicating whether the agent receives 'Cheese', 'Shock', or 'Null'.

            ### Rules for Filling The observation matrix A:
            1. `A[0]` should be an identity matrix, where A[0][i, i, :, :] = 1 for all i from 0 to 34.
            2. `A[1]`: 
            - Default to 'Null' everywhere (A[1][0, :, :, :] = 1.0).
            - At Cue 1 location (2, 3), provide signals for Cue 2's location:
                - A[1][1, 17, 0, :] = 1.0, A[1][2, 17, 1, :] = 1.0, A[1][3, 17, 2, :] = 1.0, A[1][4, 17, 3, :] = 1.0.
            3. `A[2]`: 
            - Default to 'Null' everywhere (A[2][0, :, :, :] = 1.0).
            - At each Cue 2 location, fill according to reward conditions:
                - A[2][1, <Cue 2 index>, i, 0] = 1.0 for 'TOP'; A[2][2, <Cue 2 index>, i, 1] = 1.0 for 'BOTTOM'.
            4. `A[3]`: 
            - Default to 'Null' everywhere (A[3][0, :, :, :] = 1.0).
            - At reward locations:
                - A[3][1, top_idx, :, 0] = 1.0 ('Cheese' at 'TOP'), A[3][2, top_idx, :, 1] = 1.0 ('Shock' at 'BOTTOM'),
                - A[3][1, bottom_idx, :, 1] = 1.0 ('Cheese' at 'BOTTOM'), A[3][2, bottom_idx, :, 0] = 1.0 ('Shock' at 'TOP').

            ### Task:
            Please generate the observation matrix (`A` array) according to these guidelines and provide the matrices in Python NumPy array format.
            Return only the JSON object with no additional text or formatting:
            {{
                \"position\": \"(y, x) which is current agent position\",
                \"next_action\": \"You should inference for the next action. Remember to compare your current location and the location of cue_1, cue_2, cheese and shock given by Human. Then, tell the user about the next action - MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY))\",
                \"action_reason\": \"explain why you perform above ation and also compare the location of cue_1, cue_2, cheese and shock given by Human to verify the explaination you gave. If you are going to reveal a location, please give the reason why.\",
                \"next_position\": \"(y, x) which is next agent position after reasoning and performing the next_action \",
                \"current_goal\": \"the name of agent's current goal such as  cue_1, L1, L2, L3, L4 or cheese\",
                \"cheese_location\": \"the location of cheese in format (y, x) or Null if you don't know where it is\",
                \"shock_location\": \"the location of shock in format (y, x) or Null if you don't know where it is\",
                \"location_obs_matrix\": \"full version of the first slice of `A` array\",
                \"cue1_obs_matrix\": \"full version of the second slice of `A` array\",
                \"cue2_obs_matrix\": \"full version of the third slice of `A` array\",
                \"reward_obs_matrix\": \"full version of the fourth slice of `A` array\",
            }}

            FOLLOW THIS INSTRUCTION BELOW AS AN SAMPLE ONLY, YOU HAVE TO USE THE INFORMATION COMES FROM ENVIRONMENT SETUP AND HUMAN MESSAGE FOR REASONING PROCESS.
            THE INFORMATION GIVEN BELOW IS JUST A MOCK EXPERIMENT, YOU MUST NOT USE THE INFORMATION GIVEN IN THE FOLLOWING QUOTES FOR REASONING PROCESS:
                \"The Grid world Dimension is: {self.env.grid_world_dimension}
                    Current location of agent is (1, 4),
                    You have to inference to move to the location of CUE 1 at (2, 1) (SAMPLE ONLY, NOT THE REAL CUE 1 LOCATION). 
                    Then there are four additional locations that will serve as possible locations for CUE 2 (SAMPLE ONLY, NOT THE REAL CUE 2 LOCATIONS) which are {{\"L1"\: (1, 3), \"L2"\: (2, 4), \"L3"\: (4, 4), \"L4"\: (5, 3)}}.
                    The one is revealed are L3 and need to reach it.
                    After reaching it, there are new informations which are reward conditions named which are {{\"FIRST"\: (2, 0), \"SECOND"\: (2, 3)}} and the one is revealed as CHEESE is FIRST so that SECOND is SHOCK.
                    You need to infering and reach CHEESE on (2, 0) (SAMPLE ONLY, NOT THE REAL CHEESE LOCATION) while avoiding SHOCK on (2, 3) (SAMPLE ONLY, NOT THE REAL SHOCK LOCATION).

                    The SAMPLE output should be: {{
                        \"position\": \"(1, 4)\",
                        \"next_action\": \"MOVE_DOWN\",
                        \"action_reason\": \"Because cue_1 is on (2, 1), perform MOVE_DOWN to move downward one cell to have the same horizontal axe with cue_1 (2, 4)\",
                        \"next_position\": \"(2, 4)\",
                        \"current_goal\": \"cue_1\",                        
                        \"cheese_location\": \"Null\"
                        \"shock_location\": \"Null\",
                        \"location_obs_matrix\": \"full version of the first slice of `A` array\",
                        \"cue1_obs_matrix\": \"full version of the second slice of `A` array\",
                        \"cue2_obs_matrix\": \"full version of the third slice of `A` array\",
                        \"reward_obs_matrix\": \"full version of the fourth slice of `A` array\",
                    }}
                \"
            You are tasked with constructing an observation matrix for an agent performing active inference in a 5x7 grid world. The agent must navigate this grid to find a hidden reward ('Cheese') while avoiding a punishment ('Shock').
            TRY TO GENERATE LEAST TOKENS AS YOU CAN TOO IMPROVE PERFORMANCE BUT STILL KEEP THE SAME REASONING METHOD FOR YOUR ACTION.
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
        
        if llm_obs['reset'] == True:
            self.reset()
        else:
            obs_message = f'The agent is now at {llm_obs['position']} and about to {llm_obs['next_action']} to {llm_obs['next_position']} in order to find {llm_obs['current_goal']}. Please move to {llm_obs['next_position']} and infer for what action should be taken to further reduce uncertainty and find the current goal. If you are trying to reach cheese after reaching cue 2 and shock is on the way. Find another path.'

            if llm_obs['position'] == str(self.env.cue_1_location):
                self.message_history.append(SystemMessage(content='WHAT IS CUE 2 NAME?'))
                obs_message = f"These are cue 2 possible locations: {{\"L1\": {self.env.cue_2_locations[0]}, \"L2\": {self.env.cue_2_locations[1]}, \"L3\": {self.env.cue_2_locations[2]}, \"L4\": {self.env.cue_2_locations[3]}}} and the one specified as cue_2 is {self.env.cue_1_obs}, the other locations are empty now. Keep Infering until reaching {self.env.cue_1_obs}. "
                
                # Cue 2 location in 1D array
                self.cue_2_1d_idx = self.env.cue_2_location[1]*self.env.column+self.env.cue_2_location[0]
                obs_message += f"\nPlease fill the matrix A[]"

            if llm_obs['position'] == str(self.env.cue_2_location):
                self.message_history.append(SystemMessage(content='WHAT IS REWARD CONDITION?'))
                obs_message = f"These are possible reward locations: {{\"{self.env.reward_conditions[0]}\": {self.env.reward_locations[0]}, \"{self.env.reward_conditions[1]}\": {self.env.reward_locations[1]}}} and the CHEESE is {self.env.cue_2_obs} so that the SHOCK is the other one in possible reward locations is SHOCK. Keep Infering until reaching the CHEESE and try to avoid the SHOCK"
                
                # Reward location in 1D array
                self.reward_1d_idx = self.env.reward_1d_idx[1]*self.env.column+self.env.reward_1d_idx[0]
            
            self.message_history.append(HumanMessage(content=obs_message))
            self.log_conversation(obs_message)
            
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
    