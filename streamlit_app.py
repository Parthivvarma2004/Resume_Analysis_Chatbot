import streamlit as st

import re
import fitz

import os
from dotenv import load_dotenv

from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import time

if os.name == 'posix' and os.uname().sysname == 'Linux':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import sqlite3
    print(f"sqlite3 version: {sqlite3.sqlite_version}")

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# set page title
st.set_page_config(page_title='Team Byte Busters')
st.title('Welcome to our chatbot website!')

# remove new line characters and extra spaces
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# convert pdf to text
def pdf_to_text(file, start_page=1, end_page=None):
    doc = fitz.open(stream=file[0].read(), filetype='pdf')
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

# generate response from uploaded resume using openai
def generate_response(uploaded_file, openai_api_key, query_text):
    # load document if file is uploaded
    if uploaded_file is not None:
        documents=pdf_to_text(uploaded_file)
        
        # split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        
        # select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        
        # create retriever interface
        retriever = db.as_retriever()
        
        # create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key),
            chain_type='stuff',
            retriever=retriever
        )
        
        # create a progress bar
        progress_bar = st.progress(0)
        progress_percent = 0
        
        response = qa.run(query_text)
        
        # update progress bar
        progress_percent = 100
        progress_bar.progress(progress_percent)
        
        # remove progress bar
        time.sleep(0.5)
        progress_bar.empty()
        
        return response
        

uploaded_file = st.file_uploader('Upload your resume', type='pdf', accept_multiple_files=True)

result = []
with st.form('myform', clear_on_submit=False):
    query_text = st.text_input('Enter your question:', placeholder = "What is the candidate's GPA?", disabled=not uploaded_file)
    
    submitted = st.form_submit_button('Ask', disabled=not uploaded_file)
    
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Thinking...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            st.balloons() 

if len(result):
    st.info(response)

#If you have any questions, checkout our [documentation](add a link to our instruction manual here) 