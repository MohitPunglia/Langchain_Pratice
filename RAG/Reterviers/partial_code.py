# %% [markdown]
# # WikipediaRetriever Code Example
# 
# 

# %%
!pip install langchain-community wikipedia chromadb

# %%
from langchain_community.retrievers import WikipediaRetriever

# %%
retrievers=WikipediaRetriever(top_k_results=2,lang='en')
query='How many total states are there in India?'
retrievers.invoke(query)

# %% [markdown]
# # Vector Store Retriever Code Example

# %%
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings

# %%
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# %%
# Initialize open-source embeddings
embedding_function=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# %%
vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    collection_name='my_collection'
)


# %%
retriever=vector_store.as_retriever()

# %%
query='what is Chroma used for?'
results=retriever.invoke(query)

for i in results:
  print('Result ')

  print(i.page_content)



# %%
vector_store.similarity_search(query,k=2)

# %% [markdown]
# # MMR Maximum Marginal Relevance Code Example
