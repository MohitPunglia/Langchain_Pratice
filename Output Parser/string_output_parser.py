from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Prompt 1
template1 = PromptTemplate(
    template="Provide detail report about the {topic}", input_variables=["topic"]
)

# Prompt 2
template2 = PromptTemplate(
    template="provide 3 line summary for the following text:\n {text}",
    input_variables=["text"],
)

str_output_parser = StrOutputParser()

# Create chain
chain = template1 | model | str_output_parser | template2 | model | str_output_parser

result = chain.invoke({"topic": "Google Pixel 9 Pro"})

print(result)
