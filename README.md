# LangChain Practice Repository 🦜️🔗

A hands-on collection of Python scripts and modules to explore LangChain components like chat models, prompt templates, embeddings, and agents. Ideal for learning LangChain workflows and LLM integration!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
This repository contains practical examples of **LangChain** components, demonstrating how to:
- Build and customize chat models.
- Design reusable prompt templates.
- Generate text embeddings.
- Create basic LLM-powered agents.

---

## Project Structure
```
Langchain_Practice/
├── chat_model/               # Chat model implementations (OpenAI, HuggingFace, etc.)
│   ├── basic_chat.py         # Simple chat completion example
│   └── streamed_chat.py      # Streaming chat responses
├── prompt_templates/         # Predefined prompt templates
│   ├── summarization_prompt.py  # Summarization template
│   └── qa_prompt.py          # Question-answering template
├── embedding_models/         # Text embedding examples
│   └── text_embeddings.py    # Generate embeddings with OpenAI/SentenceTransformers
├── agents/                   # Agent implementations
│   └── simple_agent.py       # Basic LLM agent workflow
├── .env.example              # Template for environment variables
├── requirements.txt          # Dependencies
└── README.md
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Langchain_Practice.git
   cd Langchain_Practice
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   - Rename `.env.example` to `.env`.
   - Add your API keys (e.g., OpenAI, HuggingFace Hub):
     ```env
     OPENAI_API_KEY=your_api_key_here
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```

---

## Usage Examples

### 1. Chat Models
```python
# From chat_model/basic_chat.py
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.7)
response = chat.predict("Explain quantum computing in 2 sentences.")
print(response)
```

### 2. Prompt Templates
```python
# From prompt_templates/qa_prompt.py
from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context:
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
```

### 3. Embeddings
```python
# From embedding_models/text_embeddings.py
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "LangChain simplifies LLM application development."
embedding = embeddings.embed_query(text)
```

### 4. Agents
```python
# From agents/simple_agent.py
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("What's the weather in Tokyo today?")
```

---

## Configuration
Ensure your `.env` file includes these keys (if using corresponding services):
```env
OPENAI_API_KEY=your_key_here           # For OpenAI models
HUGGINGFACEHUB_API_TOKEN=your_token    # For HuggingFace Hub models
SERPAPI_API_KEY=your_key_here          # For SerpAPI (agent example)
```

---

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a branch: `git checkout -b feature/new-example`.
3. Add/modify examples (e.g., new agents, chains, or tools).
4. Test your code and update documentation if needed.
5. Submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or feedback:
- GitHub: https://github.com/MohitPunglia
