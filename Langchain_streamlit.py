from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from io import BytesIO

load_dotenv()
os.getenv("OPENAI_API_KEY")

api_key=os.getenv('OPENAI_API_KEY')

model=ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=api_key)

st.title('ICT Bill Chatbot')
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

files = st.file_uploader(
    "Upload pdf files",
    accept_multiple_files=True,
    key=st.session_state["file_uploader_key"],
    type=["pdf"],
)

if files:
    st.session_state["uploaded_files"] = files
    lst=[]

    for file in files:
        if file.type == "application/pdf":
            pdf = PdfReader(BytesIO(file.getvalue()))
            text = ''
            for page in range(len(pdf.pages)):
                text += pdf.pages[page].extract_text()
            df = pd.DataFrame([text], columns=["Text"])
            lst.append(df)
        
    agent=create_pandas_dataframe_agent(model,lst,verbose=False,allow_dangerous_code=True)

if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()



if 'last_question' not in st.session_state:
    st.session_state.last_question=''
    
if 'show_response' not in st.session_state:
    st.session_state.show_response=False

user_input=st.text_input('Ask your question',value='',key='user-query')
submit_button=st.button('Ask')

if submit_button:
    st.session_state.last_question=user_input
    st.session_state.show_response=True
    if user_input.lower()!='quit':
        response=agent.run(user_input)
        st.session_state.response=response
        
    else:
        st.stop()
        
st.header('Respose',divider='rainbow')

if 'response' not in st.session_state:
    st.session_state.response='No response yet'

if st.session_state.show_response:
    st.markdown(f'<p style="background-color:lightblue; padding: 8px 8px; border-radius: 5px;">Question: {st.session_state.last_question}</p>', unsafe_allow_html=True)
    st.write(f"Answer: {st.session_state.response}")
    st.session_state.show_response = False  
