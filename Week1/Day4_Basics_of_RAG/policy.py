from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.fake import FakeListLLM

# Step 1: Load document
loader = TextLoader("company_policy.txt")
documents = loader.load()

# Step 2: Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 3: Embed chunks using sentence-transformers
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4: Store in FAISS vector store
db = FAISS.from_documents(docs, embedding)

# Step 5: Setup Retriever
retriever = db.as_retriever()

# Step 6: Use a Fake LLM for demo (or plug in OpenAI/Local model)
# Replace this with an actual LLM if needed
llm = FakeListLLM(responses=["You can get a refund within 30 days if the product is unused."])
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 7: Ask your question
query = "What is the refund policy?"
answer = qa.run(query)

print(f"Q: {query}\nA: {answer}")
