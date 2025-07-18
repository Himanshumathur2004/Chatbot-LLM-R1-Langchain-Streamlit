# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
# from transformers import pipeline
# from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



 

model = ChatOpenAI(
    openai_api_key="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",       
    openai_api_base="https://openrouter.ai/api/v1", 
    model_name="tngtech/deepseek-r1t2-chimera:free",             
)


chat_history = []

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]
while True:
    user_inpout = input("User :\n")
    chat_history.append(user_inpout)  # Append user input to chat history
    if user_inpout.lower() == "exit":
        print("Exiting the chatbot.")
        break   
    
    result=model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI :",result.content)
    
    
    # load_dotenv()
    
# pipeline=pipeline(
#     "text-generation",
#     model="tngtech/deepseek-r1t2-chimera",
#     device=0
# )

# model=HuggingFacePipeline(
#     pipeline=pipeline
# )

# model= ChatHuggingFace(llm=llm)