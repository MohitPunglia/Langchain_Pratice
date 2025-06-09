from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

template = PromptTemplate(
    template="Summarize the following document:\n\n{document}",
    input_variables=["document"],
)

parser = StrOutputParser()

loader = UnstructuredFileLoader("RAG/Document_Loader/Pratice_word_doc.docx")

docs = loader.load()

# print(docs)

# print(docs[0].page_content)

chain = template | model | parser

print(chain.invoke({"document": docs[0].page_content}))
