import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------- Config ----------
st.set_page_config(page_title="DeepSeek Chatbot", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        :root {
            color-scheme: light dark;
        }
        body {
            background: linear-gradient(145deg, #f0f4ff, #e8ecf1);
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            color: var(--text-color);
            margin-bottom: 1.5rem;
        }
        .chat-message.user {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 0.8rem;
            border-radius: 10px;
            margin: 10px 0;
        }
        .chat-message.assistant {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 0.8rem;
            border-radius: 10px;
            margin: 10px 0;
        }
        .creator {
            text-align: right;
            font-style: italic;
            color: var(--text-color);
            margin-bottom: 1rem;
        }
        .stChatMessage {
            padding: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# @media (prefers-color-scheme: dark) {
#     body {
#         background: #121212;
#     }
# }
# @media (prefers-color-scheme: light) {
#     body {
#         background: linear-gradient(145deg, #f0f4ff, #e8ecf1);
#     }
# }

model = ChatOpenAI(
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1", 
    model_name="deepseek/deepseek-r1-0528:free",             
)

# ---------- Title ----------
st.markdown("<div class='title'>R1 Model</div>", unsafe_allow_html=True)
st.markdown("""
    <style>
        .creator {
            text-align: Right;
            padding-left: 10px;
            font-style: italic;
            color: #666;
        }
    </style>
    <div class='creator'>Created and designed by Himanshu</div>
""", unsafe_allow_html=True)

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# ---------- User Input ----------
user_input = st.chat_input("Type your message...")

# ---------- Display History ----------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message user'>{msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"<div class='chat-message assistant'>{msg.content}</div>", unsafe_allow_html=True)

# ---------- Handle New Message ----------
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message user'>{user_input}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response = model.invoke(st.session_state.chat_history)
        st.markdown(f"<div class='chat-message assistant'>{response.content}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append(AIMessage(content=response.content))
