
# langchain_loader.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Create a sample text file
sample_text = """
LangChain is a framework for developing applications powered by language models. 
It enables applications that are data-aware, agentic, and can use language models for reasoning.

LangChain provides integrations with many different language models, tools, and vector stores.
This makes it easier to build LLM-powered applications by handling retrieval, chaining, and memory.
"""

with open("sample.txt", "w") as file:
    file.write(sample_text)

# Step 2: Load the text file using TextLoader
loader = TextLoader("sample.txt")
documents = loader.load()

# Step 3: Split the documents using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

# Step 4: Print the number of chunks
print(f"Total number of document chunks: {len(chunks)}")
