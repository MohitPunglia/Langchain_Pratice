# -*- coding: utf-8 -*-
"""Langchain_RAG_Example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_Qpfr2KlZxCbCbUyLdonpnXqex-tLiW_
"""

!pip install -q youtube-transcript-api langchain-community faiss-cpu langchain-groq

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from google.colab import userdata

"""## Step 1a - Indexing (Document Ingestion)"""

video_id='Gfr50f6ZBvo' # only video id not the full URL

try:
  transcript_list=YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=['en'])

  transcript = " ".join(chunk["text"] for chunk in transcript_list)
  print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

transcript_list

"""## Step 1b - Indexing (Text Splitting)"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=250
)

chunks=splitter.create_documents([transcript])

len(chunks)

chunks[79]

"""# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)"""

# Initialize open-source embeddings
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.index_to_docstore_id

vector_store.get_by_ids(['c1241e51-a9f5-41fb-9c44-3771f53be391'])

"""##Step 2 - Retrieval"""

retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever.invoke('What is deepmind')

"""##Step 3 - Augmentation"""

llm=ChatGroq(model_name='llama3-70b-8192',api_key=userdata.get('GROQ_API_KEY'))

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
retrieved_docs