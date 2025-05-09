from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# Use Zephyr-7B-beta instead (confirmed working)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)


class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
The Google Pixel 9 Pro is a fantastic phone that delivers a premium experience, particularly for those who value software and camera performance. While the price is a consideration, the phone offers a compelling blend of features and performance.""")

print(result)
