from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

load_dotenv()


# Define schema
class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Return sentiment of the review"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="Write down all the pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="Write down all the cons inside a list"
    )
    name: Optional[str] = Field(default=None, description="Name of the reviewer")


# Create JSON parser
parser = JsonOutputParser(pydantic_object=Review)

# Build prompt template
template = """<|system|>
Analyze this product review and format your response as JSON with the following structure:
{format_instructions}

Follow these rules:
1. Always return valid JSON
2. Use double quotes for strings
3. Don't include markdown formatting</s>
<|user|>
{input}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template).partial(
    format_instructions=parser.get_format_instructions()
)

# Initialize model without structured output
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

# Create chain
chain = prompt | llm | parser

# Run analysis
result = chain.invoke(
    {
        "input": """I recently upgraded to the Samsung Galaxy S24 Ultra... (your review text here)"""
    }
)

# Convert to Pydantic model
review = Review(**result)
print(review)
