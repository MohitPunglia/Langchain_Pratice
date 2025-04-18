from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define a list of documents
documents = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document.",
    "This is the fourth document.",
]

query = " 1st document is the first document."
# Generate embedding for the query
vector = embedding.embed_query(query)

# Generate embeddings for the documents
embeddings = embedding.embed_documents(documents)

# Print the embeddings
print(embeddings)

# Print the query embedding
print("Query Embedding:")
print(vector)
# Calculate cosine similarity between the query and the documents
similarity_matrix = cosine_similarity([vector], embeddings)
# Print the similarity scores
print("Cosine Similarity Scores:")
for i, score in enumerate(similarity_matrix[0]):
    print(f"Query vs Document {i + 1}: {score:.4f}")
# The above code is an example of how to use the HuggingFaceEmbeddings class from the langchain_huggingface module to generate embeddings for a list of documents and a query, and then calculate cosine similarity between them.
# The code first imports the HuggingFaceEmbeddings class and cosine_similarity function, then creates an instance of HuggingFaceEmbeddings with a specific model name.
# It then defines a list of documents and a query, and uses the embed_query method of the HuggingFaceEmbeddings instance to generate an embedding for the query.
# It also generates embeddings for the documents using the embed_documents method.
# Finally, it calculates the cosine similarity between the query and all documents using the cosine_similarity function from sklearn.metrics.pairwise and prints the resulting similarity scores.
# Now we will calculate cosine similarity between the documents.
# Calculate cosine similarity between the documents
similarity_matrix = cosine_similarity(embeddings)
# Print the similarity scores
print("Cosine Similarity Scores between Documents:")
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        print(f"Document {i + 1} vs Document {j + 1}: {similarity_matrix[i][j]:.4f}")
# The above code is an example of how to calculate cosine similarity between a list of documents using the HuggingFaceEmbeddings class from the langchain_huggingface module.
# The code first imports the cosine_similarity function from sklearn.metrics.pairwise.
# It then generates embeddings for a list of documents using the embed_documents method of the HuggingFaceEmbeddings instance.
# After generating the embeddings, it calculates the cosine similarity between all pairs of documents using the cosine_similarity function.
# Finally, it prints the resulting similarity scores for each pair of documents.
# Now we will calculate cosine similarity between the first document and the others.
