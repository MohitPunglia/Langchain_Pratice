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