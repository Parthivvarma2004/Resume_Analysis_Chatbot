import streamlit as st

import re
import fitz

import os
from dotenv import load_dotenv
import openai

from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import asyncio
from pgml import Database

import psycopg2

import time
import random
import base64

import pdf2image
import zipfile
import io

from streamlit_image_select import image_select

COLLECTION_NAME = "resumes"

if os.name == 'posix' and os.uname().sysname == 'Linux':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import sqlite3
    print(f"sqlite3 version: {sqlite3.sqlite_version}")
    
load_dotenv()
    
# setting up the database
conninfo = os.environ['DATABASE_URL']
db = Database(conninfo)

openai_api_key = os.environ['OPENAI_API_KEY']
openai.api_key = openai_api_key

# set page title
st.set_page_config(page_title='Team Byte Busters')
st.title('Welcome to our chatbot website!')

# remove new line characters and extra spaces
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# convert pdf to text
#def pdf_to_text(files):
 #   text_list = []
  #  for file in files:
   #     doc = fitz.open(stream=file.read(), filetype='pdf')
    #    total_pages = doc.page_count
#
 #       for i in range(total_pages):
  #          text = doc.load_page(i).get_text("text")
   #         text = preprocess(text)
    #        text_list.append(text)
#
 #       doc.close()
  #  return text_list

# convert pdf to text
def pdfs_to_documents(files):
    documents = []
    for file in files:
        doc = fitz.open(stream=file.read(), filetype='pdf')
        total_pages = doc.page_count

        for i in range(total_pages):
            text = doc.load_page(i).get_text("text")
            text = preprocess(text)
            documents.append({"text":text,"page number": i, "source":file.name})
        doc.close()
    
    return documents

# generate response from uploaded resume using openai
#def generate_response(uploaded_files, openai_api_key, query_text):
    # load document if file is uploaded
 #   if uploaded_files is not None:
  #      documents=pdf_to_text(uploaded_files)
   #     
        # split documents into chunks
    #    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
     #   texts = text_splitter.create_documents(documents)
        
        # select embeddings
      #  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # create a vectorstore from documents
       # db = Chroma.from_documents(texts, embeddings)
        
        # create retriever interface
        #retriever = db.as_retriever()
        
        # create QA chain
        #qa = RetrievalQA.from_chain_type(
        #    llm=ChatOpenAI(openai_api_key=openai_api_key),
        #    chain_type='stuff',
        #    retriever=retriever
        #)
        
        # create a progress bar
        #progress_bar = st.progress(0)
        #progress_percent = 0
        
        #response = qa.run(query_text)
        
        # update progress bar
        #progress_percent = 100
        #progress_bar.progress(progress_percent)
        
        # remove progress bar
        #time.sleep(0.5)
        #progress_bar.empty()
        
        #return response
    
def generate_response(context_for_resume):
    messages = [  
                {'role':'system',
                'content':'You are a resume analyzer. Resume text will be given to you and you must find relevant information about the candidates from them.'},    
                {'role':'user', 
                'content':context_for_resume},  
                ] 
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = 0
    )
    return response["choices"][0]["message"]["content"]
    
async def database_functions(collection_name, documents, db):
    collection = await db.create_or_get_collection(collection_name)
    await collection.upsert_documents(documents)
    
    #generating chunks and embeddings
    await collection.generate_chunks()
    await collection.generate_embeddings()

async def vector_search_function(collection_name, query_text, db):
    collection = await db.create_or_get_collection(collection_name)
    vector_search_results = await collection.vector_search(query_text, top_k = 3)
    context = ""
    for search_result in vector_search_results:
        context += search_result[1] + "/n"
    context += query_text
    return context   

async def delete_all_data(db, collection_name):
    db.archive_collection(collection_name)

#Opening note

st.title("Important notice 📄")

with st.chat_message("assistant"):
    
    st.write("🚀 **For a better user experience and to avoid any confusion,** we kindly request all users testing our app to clear the database before use.")

    st.write("🔴 **Clearing the database before use will ensure that the chatbot doesn't mix up your data with previous users,** allowing you to have a smooth experience during testing.")

    st.write("Thank you for your cooperation!")

#delete button

st.title("Clear database")

with st.chat_message("assistant"):
    st.write("**Click the button below to delete all data from the database.**")
    confirmation = st.checkbox("I understand that this action will delete all data. Confirm?")
    if st.button("Delete All Data") and confirmation:
        with st.spinner('Deleting files from database...'):
            asyncio.run(delete_all_data(db=db, collection_name=COLLECTION_NAME))
            st.success('Database cleared!', icon="🗑️")
    
with st.chat_message("assistant"):
    with st.form('FileUploadForm', clear_on_submit=False):
        uploaded_files = st.file_uploader('Upload your resume', type='pdf', accept_multiple_files=True)
        add_resume_to_database = st.form_submit_button('Add to database')
        
        if add_resume_to_database:
            with st.spinner('Adding files to database...'):
                uploaded_documents = []
                uploaded_documents = pdfs_to_documents(uploaded_files)
                asyncio.run(database_functions(collection_name = COLLECTION_NAME, documents = uploaded_documents, db=db))
                st.success('Files Added!', icon="✅")

# result = []
# with st.form('Queryform', clear_on_submit=False):
#     query_text = st.text_input('Enter your question:', placeholder = "Ask a question to get information on the resumes in our database")
    
#     submitted = st.form_submit_button('Ask')
    
#     if submitted and openai_api_key.startswith('sk-'):
        
#         with st.spinner('Thinking...'):
#             context_for_resume = asyncio.run(vector_search_function(COLLECTION_NAME, query_text, db))
#             #print(context_for_resume)
#             response = generate_response(context_for_resume)
#             result.append(response)
#             st.success('Query received!', icon="✅")

# if len(result):
#     st.info(response)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# NOT WORKING SESSION STATE MANAGEMENT
# also just repeats current messages after uploading new resume in middle of chat session

# if "messages" in st.session_state:
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.write(f"User: {message['content']}")
#         else:
#             st.write(f"Assistant: {message['content']}")
        
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query_text := st.chat_input("Ask a question to get information on the resumes in our database"):
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner('Thinking...'):
            try:
                context_for_resume = asyncio.run(vector_search_function(COLLECTION_NAME, query_text, db))
                #print(context_for_resume)
    
                response = generate_response(context_for_resume)
            except Exception as e: 
                response = "Currently, our database does not contain any resumes. We kindly request you to add a resume to our database before proceeding with any questions for the chatbot. Thank you for your cooperation and understanding."
        
        for chunk in response.split():
            full_response += chunk + " "
            sleep_time = random.triangular(0.005, 0.06, 0.0001)
            time.sleep(sleep_time)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
                                
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
#If you have any questions, checkout our [documentation](add a link to our instruction manual here) 