from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class APIDocumentation(BaseModel):
    endpoint: str = Field(description="API endpoint")
    method: str = Field(description="HTTP method (GET, POST, etc.)")
    description: str = Field(description="Brief description of the API")
    parameters: dict = Field(description="Parameters accepted by the API")
    request: dict = Field(description="Request body format")
    response: dict = Field(description="Response body format")


parser = PydanticOutputParser(pydantic_object=APIDocumentation)

template = PromptTemplate(
    template="Generate API documentation for the following endpoint: {endpoint}",
    input_variables=["endpoint"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"endpoint": "https://api.example.com/v1/resource"})

print(result)
# This code generates API documentation for a given endpoint using a language model and Pydantic for structured output parsing.
# It defines a Pydantic model for the API documentation structure, creates a prompt template, and invokes the model to generate the documentation.
# The result is printed at the end.
# This code is useful for generating structured API documentation dynamically based on user input or predefined templates.
# It can be extended to include more fields or different structures as needed.
# The use of Pydantic ensures that the output adheres to the defined schema, making it easier to work with and validate.
# The code is well-structured and follows best practices for using language models and structured output parsing.
