import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import tempfile
import pandas as pd

# --- Google Fonts and Custom CSS for Modern UI ---
st.markdown('''
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .main {
            background-color: #ffffffcc;
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }
        .stButton>button {
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: bold;
            transition: box-shadow 0.2s, transform 0.2s;
            box-shadow: 0 2px 8px rgba(0,114,255,0.10);
        }
        .stButton>button:hover {
            box-shadow: 0 4px 16px rgba(0,114,255,0.18);
            transform: translateY(-2px) scale(1.03);
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        .stSidebar {
            background: #e0e7ef;
        }
        .stChatMessage {
            border-radius: 12px;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .gradient-banner {
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border-radius: 18px;
            padding: 1.5rem 1rem 1rem 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 24px rgba(0,114,255,0.10);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .banner-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
        }
        .banner-logo {
            width: 60px;
            height: 60px;
        }
        .card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
        }
        .chat-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 0.7rem;
            border: 2px solid #0072ff22;
        }
        .chat-row {
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.7rem;
        }
        .chat-bubble {
            background: #f0f6ff;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            box-shadow: 0 1px 4px rgba(0,114,255,0.04);
            max-width: 80%;
        }
        .chat-bubble.assistant {
            background: #e6f3ff;
        }
        .chat-bubble.user {
            background: #d1ffe6;
        }
    </style>
''', unsafe_allow_html=True)

# --- Gradient Banner with Logo and App Name ---
st.markdown('''
    <div class="gradient-banner">
        <img src="https://img.icons8.com/color/96/000000/sql.png" class="banner-logo"/>
        <span class="banner-title">🦜 LangChain: Chat with SQL DB</span>
    </div>
''', unsafe_allow_html=True)

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

# --- Sidebar organization ---
st.sidebar.header("Database Connection")
radio_opt=["Use SQLLite 3 Database- Student.db","Connect to you MySQL Database","Upload Excel or CSV File"]
selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Host")
    mysql_user=st.sidebar.text_input("MYSQL User")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MySQL database")
elif radio_opt.index(selected_opt)==2:
    db_uri="EXCEL_UPLOAD"
    uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
    excel_table_name = st.sidebar.text_input("Table name for uploaded data", value="excel_data")
    if uploaded_file is None:
        st.warning("Please upload an Excel or CSV file to continue.")
else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="GRoq API Key",type="password")

if not db_uri:
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add the groq api key")

## LLM model
llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None, uploaded_file=None, excel_table_name=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   
    elif db_uri=="EXCEL_UPLOAD":
        if uploaded_file is None:
            st.error("Please upload an Excel or CSV file.")
            st.stop()
        # Read file into DataFrame
        file_name = getattr(uploaded_file, 'name', '').lower() if uploaded_file is not None else ''
        if file_name.endswith('.csv') and uploaded_file is not None:
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        elif uploaded_file is not None:
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            st.error("No file uploaded.")
            st.stop()
        # Show preview of uploaded data
        with st.expander("Preview Uploaded Data", expanded=True):
            st.dataframe(df.head(20))
        # Create a temporary SQLite DB
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(temp_db.name)
        table_name = excel_table_name if isinstance(excel_table_name, str) and excel_table_name else "excel_data"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        creator = lambda: sqlite3.connect(temp_db.name)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
elif db_uri=="EXCEL_UPLOAD":
    if uploaded_file is not None:
        db=configure_db(db_uri, uploaded_file=uploaded_file, excel_table_name=excel_table_name)
    else:
        db = None
else:
    db=configure_db(db_uri)

# --- Main chat area layout ---
with st.container():
    st.markdown("<h3 style='color:#0072ff;'>💬 Chat Interface</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #e0e7ef;'>", unsafe_allow_html=True)

    if db is not None:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
    else:
        toolkit = None
        agent = None

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # --- Card-style chat area ---
    with st.container():
        for msg in st.session_state.messages:
            avatar_url = "https://img.icons8.com/color/48/000000/bot.png" if msg["role"] == "assistant" else "https://img.icons8.com/color/48/000000/user-male-circle--v2.png"
            bubble_class = "assistant" if msg["role"] == "assistant" else "user"
            st.markdown(f'''
                <div class="chat-row">
                    <img src="{avatar_url}" class="chat-avatar"/>
                    <div class="chat-bubble {bubble_class}">{msg["content"]}</div>
                </div>
            ''', unsafe_allow_html=True)

    user_query = st.chat_input(placeholder="Ask anything from the database") if agent is not None else None

    if user_query and agent is not None:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.markdown(f'''
            <div class="chat-row">
                <img src="https://img.icons8.com/color/48/000000/user-male-circle--v2.png" class="chat-avatar"/>
                <div class="chat-bubble user">{user_query}</div>
            </div>
        ''', unsafe_allow_html=True)

        with st.spinner('Thinking...'):
            with st.container():
                streamlit_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(user_query, callbacks=[streamlit_callback])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f'''
                    <div class="chat-row">
                        <img src="https://img.icons8.com/color/48/000000/bot.png" class="chat-avatar"/>
                        <div class="chat-bubble assistant">{response}</div>
                    </div>
                ''', unsafe_allow_html=True)

        


