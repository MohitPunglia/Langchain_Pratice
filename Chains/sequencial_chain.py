from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Provide detail report about the {topic}", input_variables=["topic"]
)

str_output_parser = StrOutputParser()

template2 = PromptTemplate(
    template="provide 3 line summary for the following text:\n {text}",
    input_variables=["text"],
)
# Create chain
chain = template1 | model | str_output_parser | template2 | model | str_output_parser

result = chain.invoke({"topic": "Trending AI news and updates"})

print(result)

print(chain.get_graph().draw_mermaid())
