import json
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama

# Load JSON dataset
json_file_path = "Expertise.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

questions = [item["question"] for item in dataset]
answers = [item["answer"] for item in dataset]

# Initialize ChromaDB
chroma_path = "chroma_db"
chroma_client = chromadb.PersistentClient(path=chroma_path)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 256}
)

# Store data in ChromaDB
vector_store = Chroma.from_texts(
    texts=questions,
    metadatas=[{"answer": ans} for ans in answers],
    embedding=embeddings,
    persist_directory=chroma_path
)
print("Dataset successfully stored in ChromaDB!")

# Function to retrieve response from ChromaDB
def retrieve_response(query):
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].metadata["answer"] if docs else "Sorry, I don't know the answer."

# Load LLaMA model for final answer generation
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q2_K.gguf",
            verbose=False)


# RAG-based chatbot response(Retrieve + Generate)
def rag_chatbot(query):
    retrieved_text = retrieve_response(query)

    prompt = (
    f"User asked: {query}\n\n"
    f"Relevant Info: {retrieved_text}\n\n"
    f"Answer:concisely based only on the relevant info above:"
    )

    response = llm(prompt, max_tokens=100, temperature=0.7)
    return response["choices"][0]["text"].strip()

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Ask your question (or type 'exit'): ")
        if user_input.lower() == {"exit", "quit"}:
            break
        bot_reply = rag_chatbot(user_input)
        print(f"\n🤖 Bot: {bot_reply}\n")       