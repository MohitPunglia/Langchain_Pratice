from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """Artificial intelligence (AI) is the creation of computer systems capable of doing tasks that would normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making. AI comprises a variety of technologies such as machine learning, natural language processing, robotics, and computer vision.

AI has altered industries worldwide, increasing efficiency and innovation. AI is used in healthcare to help with disease diagnosis and personalised treatment regimens. Chatbots, predictive analytics, and fraud detection help to streamline business operations. AI also plays an important part in autonomous vehicles, entertainment, and education, providing solutions that were once considered science fiction.

However, the rise of AI presents challenges. Ethical considerations such as data privacy, bias, and employment displacement must be carefully considered. The long-term integration of AI requires balancing technology developments with societal benefits.

AI has enormous potential to transform the world, making lives easier and industries more efficient. As AI’s capabilities grow, responsible development and regulation are critical to ensuring it benefits humanity while following ethical standards."""


splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

chunks = splitter.split_text(text)

print(chunks[0])

print(len(chunks))

# print(chunks)
