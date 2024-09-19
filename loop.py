from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Initialize the model
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)

# Define the template
template = "{question}"
prompt_template = PromptTemplate.from_template(template)

while True:
    # Get input from the user
    question = input("Enter your question (or type 'exit' to quit): ")
    
    # Break the loop if the user types 'exit'
    if question.lower() == 'exit':
        print("Exiting the loop. Goodbye!")
        break
    
    # Format the prompt with the user's question
    formatted_prompt_template = prompt_template.format(question=question)
    
    # Get the response from the model
    response = llm.invoke(formatted_prompt_template)
    
    # Print the response
    print("Answer:", response)
