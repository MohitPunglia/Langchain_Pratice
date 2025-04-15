from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = "Macintosh HD/Users/mohit/Developer/HF_Model"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.2, max_new_tokens=512),
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is IPL?")

print(result.content)
