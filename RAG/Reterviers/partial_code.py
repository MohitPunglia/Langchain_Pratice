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
# %% [markdown]
# # Multiquery Retriever Code Example

# %%
# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# %%
# Initialize open-source embeddings
embedding_function=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_function
)


# %%
similarity_retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':5})

# %%
from langchain.retrievers.multi_query import MultiQueryRetriever

# %%
!pip install langchain-groq
from langchain_groq import ChatGroq
from google.colab import userdata

multiquery_retriever=MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={'k':5}),
    llm=ChatGroq(model_name='llama3-70b-8192',api_key=userdata.get('GROQ_API_KEY'))
)

# %%
# Query
query = "How to improve energy levels and maintain balance?"

# %%
# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)

# %%
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# %% [markdown]
# # Contextual Compression Retriever Code Example

# %%
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# %%
    ), metadata={"source": "Doc3"}),
# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""


    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# %%
# Initialize open-source embeddings
embedding_function=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
