from typing import Literal
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="sentiment of feedback"
    )


pyparser = PydanticOutputParser(pydantic_object=feedback)


parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="classify the sentiment of feedback as positive, negative :\n {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pyparser.get_format_instructions()},
)


classifier_chain = prompt1 | model | pyparser

postiveprompt = PromptTemplate(
    template="Generate a positive response for the following feedback:\n {feedback}",
    input_variables=["feedback"],
)

negativeprompt = PromptTemplate(
    template="Generate a negative response for the following feedback:\n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", postiveprompt | model | parser),
    (lambda x: x["sentiment"] == "negative", negativeprompt | model | parser),
    RunnableLambda(lambda x: "No sentiment found"),
)

chain = classifier_chain | branch_chain
print(chain.invoke({"feedback": "The smartphone is pathetic and not worth the money."}))
