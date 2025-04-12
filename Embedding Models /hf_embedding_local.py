from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test sentence for embedding."

vector = embedding.embed_query(text)

print(vector)


# The above code is a simple example of how to use the HuggingFaceEmbeddings class from the langchain_huggingface module to generate embeddings for a given text.
# The code first imports the HuggingFaceEmbeddings class, then creates an instance of it with a specific model name.
# It then defines a test sentence and uses the embed_query method of the HuggingFaceEmbeddings instance to generate an embedding for the text.
# Finally, it prints the resulting vector.

# Now  we will use embedding_documents method to generate embeddings for a list of documents.
