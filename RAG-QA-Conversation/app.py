## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import requests

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the script.")
os.environ['GROQ_API_KEY'] = groq_api_key

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.set_page_config(page_title="Conversational RAG With PDF", layout="wide", page_icon="💬")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .chat-history {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 1.5em 1em;
        margin-bottom: 1em;
        min-height: 500px;
        max-height: 70vh;
        overflow-y: auto;
        transition: box-shadow 0.2s;
    }
    .chat-history:hover {
        box-shadow: 0 4px 18px rgba(0,0,0,0.13);
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.2em;
        gap: 0.7em;
    }
    .chat-bubble {
        padding: 0.8em 1.2em;
        border-radius: 16px;
        max-width: 80%;
        font-size: 1em;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: background 0.2s;
    }
    .chat-message.user .chat-bubble {
        background: linear-gradient(90deg, #e3f2fd 60%, #bbdefb 100%);
        color: #1565c0;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-message.assistant .chat-bubble {
        background: linear-gradient(90deg, #ede7f6 60%, #d1c4e9 100%);
        color: #4527a0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #eee;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3em;
        font-weight: bold;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .section-header {
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 0.7em;
        color: #333;
        letter-spacing: 0.5px;
    }
    .clear-btn {
        background: #f44336;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.4em 1em;
        font-size: 0.95em;
        cursor: pointer;
        margin-bottom: 1em;
        transition: background 0.2s;
    }
    .clear-btn:hover {
        background: #c62828;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex;align-items:center;gap:0.7em;margin-bottom:1.5em;'>
    <span style='font-size:2.1em;'>💬</span>
    <span style='font-size:1.5em;font-weight:700;color:#4527a0;'>Conversational RAG With PDF</span>
</div>
""", unsafe_allow_html=True)
st.write("<span style='font-size:1.1em;color:#666;'>Upload PDFs and chat with their content. Enjoy a modern, classy chat experience!</span>", unsafe_allow_html=True)

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    session_id=st.text_input("Session ID",value="default_session", help="Use a unique session for each conversation.")
    if 'store' not in st.session_state:
        st.session_state.store={}

    # Layout: left for chat history, right for chat and file upload
    left_col, right_col = st.columns([1,2], gap="large")

    with right_col:
        st.markdown("<div class='section-header'>Upload PDF(s)</div>", unsafe_allow_html=True)
        uploaded_files=st.file_uploader("Choose PDF file(s)",type="pdf",accept_multiple_files=True, help="You can upload multiple PDFs.")

    # Process uploaded PDFs (same as before)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        with right_col:
            st.markdown("<div class='section-header'>Ask a Question</div>", unsafe_allow_html=True)
            # Persistent input at the bottom
            if 'user_input' not in st.session_state:
                st.session_state.user_input = ''
            def submit_question():
                st.session_state.submit = True
            user_input = st.text_input(
                "Your question:",
                value=st.session_state.user_input,
                key="user_input_box",
                placeholder="Type your question here and press Enter...",
                help="Ask anything about your uploaded PDFs.",
                on_change=submit_question
            )
            if st.session_state.get('submit', False):
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": st.session_state.user_input},
                    config={
                        "configurable": {"session_id":session_id}
                    },
                )
                st.session_state.last_response = response['answer']
                st.session_state.last_session_history = session_history.messages
                st.session_state.user_input = ''
                st.session_state.submit = False

            # Show assistant response
            if 'last_response' in st.session_state:
                st.markdown(f"<div class='chat-message assistant'><div class='avatar'>🤖</div><div class='chat-bubble'>{st.session_state.last_response}</div></div>", unsafe_allow_html=True)

    # Show chat history in the left column
    with left_col:
        st.markdown("<div class='section-header'>Chat History</div>", unsafe_allow_html=True)
        # Add clear chat button
        if st.button("Clear Chat History", key="clear_chat", help="Remove all messages from this session.", use_container_width=True):
            if session_id in st.session_state.store:
                st.session_state.store[session_id].messages.clear()
            st.session_state.last_session_history = []
            st.session_state.last_response = ''
        session_history = None
        if 'last_session_history' in st.session_state:
            session_history = st.session_state.last_session_history
        elif 'store' in st.session_state and session_id in st.session_state.store:
            session_history = st.session_state.store[session_id].messages
        if session_history:
            for msg in session_history:
                role = getattr(msg, 'type', getattr(msg, 'role', 'user'))
                role_class = 'user' if role == 'human' else 'assistant'
                avatar = '🧑' if role_class == 'user' else '🤖'
                st.markdown(f"<div class='chat-message {role_class}'><div class='avatar'>{avatar}</div><div class='chat-bubble'><b>{role.title()}:</b> {msg.content}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#aaa;'>No chat history yet.</span>", unsafe_allow_html=True)
else:
    st.warning("Please enter the GRoq API Key")










