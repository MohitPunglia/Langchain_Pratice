from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use one of these verified working models
MODEL_CHOICES = {
    "summarization": "facebook/bart-large-cnn",
    "text-generation": "gpt2",
    "text2text-generation": "google/flan-t5-base",
}

llm = HuggingFaceEndpoint(
    repo_id=MODEL_CHOICES["text2text-generation"],
    task="text2text-generation",
    max_new_tokens=150,
)

prompt = PromptTemplate(
    template="Amazing facts about {topic}", input_variables=["topic"]
)

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "China"})
print(result)
