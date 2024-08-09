import os
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define a simple prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)

# Initialize the OpenAI LLM with your API key  
llm = OpenAI(model="gpt-3.5-turbo-instruct-0914")  # Specify the OpenAI model you want to use

# Create an LLMChain with the prompt and the LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Define a function to get a response from the LLMChain
def get_response(question):
    response = llm_chain.run({"question": question})
    return response

# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    response = get_response(question)
    print("Response:", response)
