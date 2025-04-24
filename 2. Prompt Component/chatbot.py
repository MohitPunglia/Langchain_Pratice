from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

chathistory = [
    SystemMessage(
        content="You are a helpful assistant. You can answer questions, provide explanations, and assist with various tasks. Please respond in a friendly and informative manner."
    ),
]

while True:
    user_input = input("You: ")
    chathistory.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))
    print("Assistant:", result.content)

print(chathistory)
