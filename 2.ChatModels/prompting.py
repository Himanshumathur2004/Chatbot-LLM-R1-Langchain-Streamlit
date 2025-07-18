from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0
)

llm = HuggingFacePipeline(
    pipeline=pipe,
)
model=ChatHuggingFace(llm=llm)

st.header("Himanshu Chat Model")

user_input = st.text_input("Ask any general question:")



# research_papers = [
#     "Attention Is All You Need (Transformer)",
#     "Generative Adversarial Networks (GANs)",
#     "Deep Residual Learning for Image Recognition (ResNet)",
#     "Mastering the game of Go with deep neural networks and tree search (AlphaGo)",
#     "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
#     "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Vision Transformer)"
# ]

# selected_paper = st.selectbox(
#     "Select Research Paper",
#     options=research_papers,
#     index=0,  
#     help="Select the paper you want an explanation for."
# )


# detail_levels = ["Simple", "Mid-level", "Detailed"]

# selected_detail = st.selectbox(
#     "Select Detail Level",
#     options=detail_levels,
#     index=0, 
#     help="Choose how in-depth you want the explanation to be."
# )

# output_lengths = ["Small", "Medium", "Large"]

# selected_length = st.selectbox(
#     "Select Output Length",
#     options=output_lengths,
#     index=1,  
#     help="Choose the approximate length of the explanation."
# )

# template = load_prompt("research_assistant_prompt_template.json")


# prompt=template.invoke({
#     "paper_name": selected_paper,
#     "detail_level": selected_detail,
#     "output_length": selected_length
# })


if st.button("Submit"):
    result=llm.invoke(user_input)
    st.write(result.content)  # Display the result in the Streamlit app