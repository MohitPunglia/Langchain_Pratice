from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is IPL?")
result1 = model.invoke(
    "Which team has won the higest IPL tropies, along with all the team which have won atleast once or none, tell all team names?"
)


print(result.content)
print(result1.content)
