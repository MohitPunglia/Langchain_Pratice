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

 %%
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# %%
# Initialize open-source embeddings
embedding_function=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embedding_function
)


# %%
retriever=vector_store.as_retriever(
    search_type='mmr'
)

# %%
query='What is langchain'
retriever.invoke(query)