from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
# Generate embeddings for the documents
embeddings = embedding.embed_documents(documents)
# Print the embeddings

print(embeddings)
# The above code is an example of how to use the HuggingFaceEmbeddings class from the langchain_huggingface module to generate embeddings for a list of documents.
# The code first imports the HuggingFaceEmbeddings class, then creates an instance of it with a specific model name.
# It then defines a list of documents and uses the embed_documents method of the HuggingFaceEmbeddings instance to generate embeddings for each document.
# Finally, it prints the resulting embeddings for each document.
