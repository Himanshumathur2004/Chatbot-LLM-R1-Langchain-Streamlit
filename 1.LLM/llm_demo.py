from langchain_eden import eden
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(
    model="gpt-4o",
    temperature=0.7,)

result=llm.invoke("What is the capital of France?")
print(result)