from langchain.community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="")

loader = TextLoader("Pratice_word_doc.docx")

doc = loader.load()

print(doc)
