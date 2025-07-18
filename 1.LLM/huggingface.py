import os

from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace

from dotenv import load_dotenv

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # A reliable and smaller model for local testing
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,)
    
)
model=ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india?")

print(result.content)



