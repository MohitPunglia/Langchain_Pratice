from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

url = "https://www.amazon.in/Apple-iPad-10th-Generation-Display/dp/B0BJLDFNVL/ref=sr_1_4?crid=UCJMMAPDVJ1U&dib=eyJ2IjoiMSJ9.VoQgFmJsKFudRBQY2KRxuwkd-c4siBkd-wVkLDJiUa2xDzgEMURxAWKZ5BtldWQ6p0Fexf633sisBals2kfMkh-LBcs_21lW0uWyOJk_n1KS8J9wUwZrCSddAjUYXoP-cF7wDGC8dUcqx3F34pgP-DsgaJOHrNFJFo8b9G3rCy0xxzegH6CnVcz-1ZEf969Cdwu2QRzynoZIJoLqwLyrsdtjIhJPEGHSl4Nu2JUqAMk.6s8a7_AyMkURwt21f2IZSVLDqRas7kTiFOhdkTTDMC4&dib_tag=se&keywords=ipad&qid=1748258303&sprefix=ipad%2Caps%2C272&sr=8-4&th=1"

loader = WebBaseLoader(url)

template = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n{text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

document = loader.load()

# print(document[0].page_content)
chain = template | model | parser

print(
    chain.invoke(
        {
            "question": "What is the review of the product we talking about here?",
            "text": document[0].page_content,
        }
    )
)
