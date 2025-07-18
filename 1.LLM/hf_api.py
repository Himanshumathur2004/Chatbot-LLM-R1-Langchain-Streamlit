from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B", 
    task="conversational",

)
model = ChatHuggingFace(llm=llm)

result= model.invoke("What is the capital of India?")

print(result.content)