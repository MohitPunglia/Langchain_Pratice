from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/QwQ-32B", task="text-generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("summarize the attention is all you need paper in 5 lines")
print(result.content)
