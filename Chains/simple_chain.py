from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xl", task="text2text-generation")
# llm = HuggingFaceEndpoint(
#    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.2, max_new_tokens=512),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Amazing 5 facts about {topic}", input_variables=["topic"]
)

StrOutputParser = StrOutputParser()

chain = prompt | model | StrOutputParser

result = chain.invoke({"topic": "India"})
print(result)
chain.get_graph().print_ascii()
