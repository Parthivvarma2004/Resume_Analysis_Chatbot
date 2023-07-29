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

import time

COLLECTION_NAME = "resumes"

if os.name == 'posix' and os.uname().sysname == 'Linux':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import sqlite3
    print(f"sqlite3 version: {sqlite3.sqlite_version}")
    
# setting up the database
conninfo = os.environ.get("DATABASE_URL")
db = Database(conninfo)

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
        
def generate_response(openai_api_key, context_for_resume):
    openai.api_key = openai_api_key
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

# storing uploaded file
with st.form('FileUploadForm', clear_on_submit=False):
    uploaded_files = st.file_uploader('Upload your resume', type='pdf', accept_multiple_files=True)
    add_resume_to_database = st.form_submit_button('Add file(s) to database')
    
    if add_resume_to_database:
        with st.spinner('Adding files to database...'):
            uploaded_documents = []
            uploaded_documents = pdfs_to_documents(uploaded_files)
            asyncio.run(database_functions(collection_name = COLLECTION_NAME, documents = uploaded_documents, db=db))
            st.success('Files Added!', icon="✅")


result = []
with st.form('Queryform', clear_on_submit=False):
    query_text = st.text_input('Enter your question:', placeholder = "Ask a question to get information on the resumes in our database")
    
    submitted = st.form_submit_button('Ask')
    
    if submitted and openai_api_key.startswith('sk-'):
        
        with st.spinner('Thinking...'):
            context_for_resume = asyncio.run(vector_search_function(collection_name= COLLECTION_NAME, query_text, db))
            #print(context_for_resume)
            response = generate_response(openai_api_key, context_for_resume)
            result.append(response)
            st.success('Query received!', icon="✅")

if len(result):
    st.info(response)

#If you have any questions, checkout our [documentation](add a link to our instruction manual here) 