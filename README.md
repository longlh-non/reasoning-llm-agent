# LLMs Reasoning Agent in Epistemic Cue Chaining Environment

## Introduction

This is a project that use Langchain to integrate GPT-4o model with Epistemic Cue Chaining Experiment created with PyGame and Gymnassium to analyse its abilities in performing Active Inference in Epistemic Cue Chaining experiment which fits in with the POMDP framework for decision-making in conditions of uncertainty and partial observability.

## Implementation

### Overview

Below will be the main interaction stream within the environment. First, the Agent will receive setup information from the environment including the grid world dimension, starting location, and Cue 1 location. Then, it will send the information, also called observations, that it receives to its 'brain,' which is an LLM. These observation details will be processed by the LLM and the outputs in this experiment will be called states. At this point, the environment will pretend to be a Human, confirming the current and previous states to the LLM to avoid 'hallucination.' From there, the LLM 'brain' will provide information to the embodied agent including the actions to be performed and the reasons why those actions should be taken to help the embodied agent interact with the epistemic cue chaining grid world.

![Environmen overview](/image/overview.png "This is Environment overview")


### Agent

![Agent's sample output](/image/sample-output.png "This is Agent's sample output")

### Grid world

To build a grid world environment where users can directly observe the movements of the agent, alongside having preliminary observations from the information gathered from experiments to evaluate and test, using Pygame (2020) and Gymnassium (Towers et al., 2024) simultaneously is a very good idea. While Gymnassium allows for the creation of an environment suitable for testing agents with a lot of support from built-in functions, making it very easy to conduct experiments as well as collect and synchronize data, Pygame provides features that help programmers easily update these changes in the data stream onto the user interface without much effort. (pomdp-grid-world-env.py)

#### Seen grid

![Seen grid world](/image/seen-grid-01.png "This is a Seen grid world with same experiment every time.")

#### Half-Seen grid

![Half-Seen grid world](/image/half-seen-grid.png "This is a Half-Seen grid with random agent's location and cues' location")

#### Unseen grid

|![Unseen grid world](/image/unseen-grid.png "This is a Unseen grid with fully random locations.")|

### Data Collection

![Unseen grid world](/image/experiments-with-50steps.png "This is a Unseen grid with fully random locations.")

![Unseen grid world](/image/experiments-with-high-requirements.png "This is a Unseen grid with fully random locations.")


### Data plotting

**Seen results plotting**

![Half-Seen result plotting](/image/seen/result.png "This is Seen's results plotting")

![Half-Seen SR](/image/seen/sr.png "This is Seen's SR plotting")


![Half-Seen FR](/image/seen/sr.png "This is Seen's FR plotting")

**Half-Seen results plotting**

![Half-Seen result plotting](/image/half-seen/result.png "This is Half-Seen's results plotting")



![Half-Seen SR](/image/half-seen/sr.png "This is Half-Seen's SR plotting")


![Half-Seen FR](/image/half-seen/sr.png "This is Half-Seen's FR plotting")

**Unseen results plotting**

![Unseen result plotting](/image/unseen/result.png "This is Unseen's results plotting")


![Unseen SR](/image/unseen/sr.png "This is Unseen's SR plotting")



![Unseen FR](/image/unseen/sr.png "This is Unseen's FR plotting")