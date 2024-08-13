from langchain_core.agents import AgentActionMessageLog, AgentFinish
import json

## utils.py
def is_list_of_tuples(variable):
    # Check if the variable is a list
    if isinstance(variable, list):
        # Check if all elements in the list are tuples
        return all(isinstance(item, tuple) for item in variable)
    return False

def is_list_of_strings(variable):
    # Check if variable is a list
    if isinstance(variable, list):
        # Check if all elements in the list are strings
        return all(isinstance(item, str) for item in variable)
    return False


def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "ModelResponse":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

