# from langchain_groq import chat_groq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# from langchain_core import RunnableSequence

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template="Provide detail report about the {topic}", input_variables=["topic"]
)

str_output_parser = StrOutputParser()

chain = prompt | llm | str_output_parser

print(chain.invoke({"topic": "AI"}))
