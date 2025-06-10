from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("RAG/Document_Loader/L&T_Annual Meeting.pdf")

document = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, separator="")

result = splitter.split_documents(document)

print(len(result))

# print(result[0].page_content)
