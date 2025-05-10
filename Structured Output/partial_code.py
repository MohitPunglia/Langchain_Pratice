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
