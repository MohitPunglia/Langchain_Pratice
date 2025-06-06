from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("RAG/Document_Loader/L&T_Annual Meeting.pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)

print(docs[1].metadata)
