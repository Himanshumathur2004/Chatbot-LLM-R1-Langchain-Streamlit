
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",)

document=["Hello, how are you?",
          "tell me 10 interesting facts about the world.",
            "What is the capital of France?"]

vector=embeddings.embed_documents(document)

print(str(vector))