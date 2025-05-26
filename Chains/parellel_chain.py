from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFacePipeline,
)
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# llm1 = HuggingFaceEndpoint(
#     # repo_id="meta-llama/Llama-2-7b-chat-hf", task="text-generation"
#     #repo_id="HuggingFaceH4/zephyr-7b-beta",
#     #task="text-generation",

# )

llm1 = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model1 = ChatHuggingFace(llm=llm1)


# llm2 = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
# )

llm2 = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Provide short notes about the {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 3 question and answers for the following text:\n {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into single document:\n {notes}\n {quiz}",
    input_variables=["notes", "quiz"],
)

str_output_parser = StrOutputParser()

parellel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | str_output_parser,
        "quiz": prompt2 | model2 | str_output_parser,
    }
)

mergered_chain = prompt3 | model1 | str_output_parser

chain = parellel_chain | mergered_chain

text = """Generative artificial intelligence (Generative AI, GenAI,[1] or GAI) is a subfield of artificial intelligence that uses generative models to produce text, images, videos, or other forms of data.[2][3][4] These models learn the underlying patterns and structures of their training data and use them to produce new data[5][6] based on the input, which often comes in the form of natural language prompts.[7][8]

Generative AI tools have become more common since an "AI boom" in the 2020s. This boom was made possible by improvements in transformer-based deep neural networks, particularly large language models (LLMs). Major tools include chatbots such as ChatGPT, DeepSeek, Copilot, Gemini, and LLaMA; text-to-image artificial intelligence image generation systems such as Stable Diffusion, Midjourney, and DALL-E; and text-to-video AI generators such as Sora.[9][10][11][12] Technology companies developing generative AI include OpenAI, Anthropic, Microsoft, Google, DeepSeek, and Baidu.[13][14][15]

Generative AI has raised many ethical questions. It can be used for cybercrime, or to deceive or manipulate people through fake news or deepfakes.[16] Even if used ethically, it may lead to the mass replacement of human jobs.[17] The tools themselves have been criticized as violating intellectual property laws, since they are trained on and emulate copyrighted works of art.[18]

Generative AI is used across many industries. Examples include software development,[19] healthcare,[20] finance,[21] entertainment,[22] customer service, sales and marketing, art, writing, fashion and product design."""

result = chain.invoke({"topic": text})

print(result)

print(chain.get_graph().draw_mermaid())
# The code demonstrates how to create a parallel chain using LangChain with Hugging Face models.
# It uses two different models to generate notes and quiz questions based on a given text.
# The results are then merged into a single document.
# The code also includes the generation of a mermaid diagram to visualize the chain.
