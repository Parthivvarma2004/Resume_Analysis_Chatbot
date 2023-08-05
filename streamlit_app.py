import streamlit as st
import pandas as pd
import ast

import re
import fitz

import os
from dotenv import load_dotenv
import openai

from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

import openpyxl

from langchain.chains import LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
import xlsxwriter
import asyncio
from pgml import Database

import psycopg2

import time
import random
import base64

import pdf2image
import zipfile
import io

import lib_platform
if lib_platform.is_platform_windows:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit_image_select import image_select

COLLECTION_NAME = "full_resumes"
COLLECTION_NAME_SUMMARIZED= "summarized_resumes"
delimiter = "####"

if os.name == 'posix' and os.uname().sysname == 'Linux':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import sqlite3
    print(f"sqlite3 version: {sqlite3.sqlite_version}")
    
#load_dotenv()
    
# setting up the database
#conninfo = os.environ['DATABASE_URL']
#db = Database(conninfo)

#openai_api_key = os.environ['OPENAI_API_KEY']
#openai.api_key = openai_api_key

Zapier_NL_API_key = "sk-ak-zD8Ymb6mMnEL3AwNxDDGwgA5EK"
db = Database("postgres://u_jkyrhncekqwp2ik:fhnv6el1ibcvwwm@02f7e6f1-1adb-4347-835a-02c74fcccb0e.db.cloud.postgresml.org:6432/pgml_scelnd4epc0lxu4")
openai_api_key = "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"
openai.api_key = openai_api_key

#React Langchain setup
docstore=DocstoreExplorer(Wikipedia())

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup"
    )
]

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
  
def parse_chatbot_output(output_str):
    try:
        # Use ast.literal_eval() to safely parse the string into a dictionary
        candidate_data = ast.literal_eval(output_str)
        return candidate_data
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing chatbot output: {e}")
        return {}

# convert pdf to text
def pdfs_to_documents(files):
    documents = []
    summarized_documents = []
    text_to_summarize = ""
    for file in files:
        doc = fitz.open(stream=file.read(), filetype='pdf')
        total_pages = doc.page_count
        text_to_summarize = ""
        for i in range(total_pages):
            text = doc.load_page(i).get_text("text")
            text = preprocess(text)
            text_to_summarize += text + "\n" 
            documents.append({"text":text,"page number": i, "source":file.name})

        summarized_text = summarizer(text_to_summarize)
        summarized_documents.append({"text":summarized_text,"source":file.name})
        doc.close()
    
    return documents, summarized_documents

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
def summarizer(resume_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages =  [  
        {'role':'system',
        'content':'You are a resume summarizer. An entire resume will be provided to you and you must summarize it within 100 tokens without losing relevant informaiton.\
            You must keep information such as candidate gpa, candidate name, candidate email, Work Experience, technical skills, Education, Certifications and Licenses, Projects and Accomplishments, Awards and Honors and Publications and Research if it exists there in the resume'},    
        {'role':'user',
        'content':f'{resume_text}'},  
        ] ,
        temperature=0.0, # this is the degree of randomness of the model's output
        max_tokens=100, # the maximum number of tokens the model can ouptut 
    )
    #print(response.choices[0].message["content"])
    return response.choices[0].message["content"]

def categorizer(query_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages =  [  
        {'role':'system',
        'content':"You will get resume text and you should reply with just the candidate's email."},    
        {'role':'user',
        'content':f'{query_text}'},  
        ] ,
        temperature=0.0, # this is the degree of randomness of the model's output
        max_tokens=200, # the maximum number of tokens the model can ouptut 
    )
    #print(response.choices[0].message["content"])
    return response.choices[0].message["content"]

def ranker(query_text):
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages =  [  
        {'role':'system',
        'content':f"You are a resume ranker.\
                Summarized resume text of candidates from a database will be sent to you along with the job details for which they are applying for.\
                The customer query will be delimited with four hashtags,i.e. {delimiter}.\
                Step 1: {delimiter}First examine the job details. What would be the best qualifications for the given job?\
                Step 2: {delimiter}Examine the resumes of the candidates and assign each of them points according to their relative suitability to the job.\
                Points should be assigned based on the following system:\
                Relevance to Job Description: out of 10 points\
                Years of Experience: out of 8 points\
                Education and Certifications: out 6 points\
                Key Skills and Expertise: out 9 points\
                Achievements and Awards: out 7 points\
                Work History and Career Progression: out of 8 points\
                Projects and Contributions: out of 7 points\
                Soft Skills and Interpersonal Abilities: out of 6 points\
                Add the points together for each candidate\
                Step 3: {delimiter}Sort the candidates in descending order based on the points assigned.\
                Step 4: {delimiter}Generate the descending list of candidates with just their names. No other information should be provided.\
                Use the following format:\
                Step 1: {delimiter}<step 1 reasoning>\
                Step 2: {delimiter}<step 2 reasoning>\
                Step 3: {delimiter}<step 3 reasoning>\
                Step 4: {delimiter}<step 4 result>\
                Make sure to include {delimiter} to separate every step.\
                "},    
        
        {'role':'user',
        'content':f'{query_text}'},  
        ] ,
        temperature=0.0, # this is the degree of randomness of the model's output
    )
    #print(response.choices[0].message["content"])
    return (response.choices[0].message["content"]).split(delimiter)[-1].strip()
  
def get_conversation():
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature = 0.0)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                f"You are a resume analyzer. \
                Resume text from a database will be sent to you and you will be asked questions on them."
            ),
            
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=300, memory_key="chat_history", return_messages=True)
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return conversation

conversation_key = "conversation"
if conversation_key not in st.session_state:
    st.session_state[conversation_key] = get_conversation()
    
conversation = st.session_state[conversation_key]
    
async def database_functions(collection_name, collection_name_summarized, documents, summarized_documents, db):
    collection = await db.create_or_get_collection(collection_name)
    await collection.upsert_documents(documents)
    
    collection_summarized = await db.create_or_get_collection(collection_name_summarized)
    await collection_summarized.upsert_documents(summarized_documents)
    #generating chunks and embeddings
    await collection.generate_chunks()
    await collection.generate_embeddings()
    
    await collection_summarized.generate_chunks()
    await collection_summarized.generate_embeddings()

async def vector_search_function(k, collection_name, query_text, db):
    collection = await db.create_or_get_collection(collection_name)
    vector_search_results = await collection.vector_search(query_text, top_k = k)
    context = ""
    for search_result in vector_search_results:
        context += search_result[1] + "\n"
    context += query_text
    return context   

async def delete_all_data(db, collection_name, collection_name_summarized):
    db.archive_collection(collection_name)
    db.archive_collection(collection_name_summarized)

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
            asyncio.run(delete_all_data(db=db, collection_name=COLLECTION_NAME, collection_name_summarized= COLLECTION_NAME_SUMMARIZED))
            #asyncio.run(delete_all_data(db=db, collection_name=COLLECTION_NAME_SUMMARIZED))
            st.success('Database cleared!', icon="🗑️")

st.title("Add resume to database")


def on_click():
    st.session_state.user_input = ""

def show_highest_gpa():
    st.session_state.user_input = "Which candidate has the highest GPA?"
    
with st.chat_message("assistant"):
    with st.form('FileUploadForm', clear_on_submit=False):
        uploaded_files = st.file_uploader('Upload your resume', type='pdf', accept_multiple_files=True)
        add_resume_to_database = st.form_submit_button('Add to database')
        
        if add_resume_to_database:
            with st.spinner('Adding files to database...'):
                uploaded_documents = []
                summarized_resumes = []
                uploaded_documents, summarized_resumes = pdfs_to_documents(uploaded_files)
                asyncio.run(database_functions(collection_name = COLLECTION_NAME, collection_name_summarized= COLLECTION_NAME_SUMMARIZED, documents = uploaded_documents, summarized_documents=summarized_resumes, db=db))
                #asyncio.run(database_functions(collection_name = COLLECTION_NAME_SUMMARIZED, documents = summarized_resumes, db=db))
                st.success('Files Added!', icon="✅")
                
    #buttons = [
        #"Clear",
        #"Who has the highest GPA?",
        #"Test"
    #]
    
    #total_chars = sum(len(button) for button in buttons)
    #relative_widths = [len(button) / total_chars for button in buttons]
    #col1, col2, col3 = st.columns(relative_widths)
        
    #with col1: st.button(buttons[0], on_click=on_click)
    #with col2: st.button(buttons[1], on_click=show_highest_gpa)
    #with col3: st.button(buttons[2], on_click=show_highest_gpa)

st.title("Compare resumes")

with st.chat_message("assistant"):
    st.info('Please note, our current database limit for comparisons is 15 resumes.')
    Job_requirement = st.text_input("Please provide the job requirements for the position that the candidates will be evaluated and ranked against.")
    is_input_given = (Job_requirement.strip() != "")
    # Button to process the input
    if st.button("Compare resumes", disabled=not is_input_given):
        with st.spinner('Comparing...'):
            query_text = f"Rank the top candidates for {Job_requirement}. If the job requirement does not have enough details for you to make a solid decision, ask the user to provide a more detailed job description. This is the maximum candidate information you can get, so ask for more job details. If the job description is not related to any of the candidate's work experience, let the user know that none of the candidates have enough experience for the particular job."
            context_for_summarized_resume = asyncio.run(vector_search_function(15, COLLECTION_NAME_SUMMARIZED, query_text, db))
            llm = OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo")
            react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
            try:
                response = react.run(context_for_summarized_resume)
            except ValueError as e:
                response = str(e)
                if not response.startswith("Could not parse LLM Output: "):
                   raise e
                response = response.removeprefix("Could not parse LLM Output: ").removesuffix("`")
            #response = ranker(query_text=context_for_summarized_resume)
            st.write(response)

st.title("Generate comparison spreadsheet")
with st.chat_message("assistant"):
    st.write("Generate a downloadable spreadsheet with key data about each candidate, including projects, work experience, skills, and more, for easy comparison.")
    if st.button("Generate spreadsheet"):
        with st.spinner('Generating...'):
            query_text = '''
            Extract the following information from the summary of each candidate's resume: skills, experiences, projects, degree. If a field is not applicable to a candidate, write that as N/A.
            Please provide candidate information in the following format:
            "Name of candidate 1": {
                "Skills": "Python, Data Analysis, Machine Learning",
                "Experiences": "Data Analyst at XYZ Company",
                "Projects": "Implemented a sentiment analysis model for customer reviews.",
                "Degree": "Bachelor of Science in Computer Science"
            },
            "Name of candidate 2": {
                "Skills": "Java, Software Development, Testing",
                "Experiences": "Software Engineer at ABC Solutions",
                "Projects": "Led the development of an e-commerce platform.",
                "Degree": "Bachelor of Engineering in Computer Engineering"
            },
            ...and so on for each candidate.
            '''
            context_for_summarized_resume = asyncio.run(vector_search_function(15, COLLECTION_NAME_SUMMARIZED, query_text, db))
            llm = OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo")
            react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
            try:
                response = react.run(context_for_summarized_resume)
            except ValueError as e:
                response = str(e)
                if not response.startswith("Could not parse LLM Output: "):
                   raise e
                response = response.removeprefix("Could not parse LLM Output: ").removesuffix("`")
                #response = response.replace("'", "\"")
                response = "{" + response + "}"
                #st.write(response)
                print(response)
                response = parse_chatbot_output(response)
                
                df = pd.DataFrame.from_dict(response, orient="index")

                # Display the DataFrame in the Streamlit app
                st.dataframe(df)
                excel_filename = "candidate_information.xlsx"
                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index_label="Candidate Name", sheet_name="Sheet1", na_rep="N/A")

                    # Save the Excel file to the buffer before closing the ExcelWriter
                    buffer.seek(0)
                    data = buffer.getvalue()

                # Create a download button for the Excel file
                st.download_button(
                    label="Download Excel File",
                    data=data,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

st.title("Schedule interview with candidate")
with st.chat_message("assistant"):
    candidate_name = st.text_input("Candidate Name")

    # Get interview date
    interview_date = st.text_input("Interview Date")

    # Get interview time
    interview_time = st.text_input("Interview Time")

    # Get interview duration
    interview_duration = st.text_input("Interview Duration")
    
    company_name = st.text_input("Hiring company")
    with st.spinner("Scheduling meeting..."):
        if st.button("Schedule meeting"):
            llm = OpenAI(openai_api_key= openai_api_key, temperature=0)
            zapier = ZapierNLAWrapper(zapier_nla_api_key= Zapier_NL_API_key)
            toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
            context_for_interview = asyncio.run(vector_search_function(1, COLLECTION_NAME, f"What is {candidate_name}'s email?", db))
            candidate_email = categorizer(context_for_interview)
            agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)
            agent.run(f"Create a new google calendar detailed event where the calendar is the hiring manager calendar. The attendees are {candidate_email}. The start date and time are {interview_date} and its from {interview_time} for {interview_duration}. The description is something like: Dear {candidate_name},\
            I hope this email finds you well. I am delighted to extend my heartfelt congratulations for being selected as one of the top candidates for the job at {company_name}.\
            Your impressive qualifications and accomplishments have truly caught our attention, and we believe you possess the skills and expertise that perfectly align with our team's requirements. We are excited to learn more about you and discuss how you can contribute to our dynamic organization.\
            We are pleased to invite you for an interview on {interview_date} at {interview_time}. The interview is expected to last approximately {interview_duration}.\
            Please let us know if the proposed time works for you. If there is any scheduling conflict, kindly reach out to us, and we will be more than happy to accommodate any adjustments.")
            st.success("Interview Scheduled!", icon="✅")
            #response = ranker(query_text=context_for_summarized_resume)
            #st.write(response)       
#st.title("Chatbot instructions") 
#with st.chat_message("assistant"):
#    st.title()

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
    user_input = st.chat_message("user")
    with user_input:
        st.markdown(query_text)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner('Thinking...'):
            try:
                category = categorizer(query_text=query_text)
                context_for_resume = asyncio.run(vector_search_function(3, COLLECTION_NAME, query_text, db))
                #print(context_for_resume)

                conversation = st.session_state[conversation_key]
                #response = conversation.predict(input = context_for_resume)
                llm = OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo")
                react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
                try:
                    response = react.run(context_for_resume)
                except ValueError as e:
                    response = str(e)
                    if not response.startswith("Could not parse LLM Output: "):
                        raise e
                    response = response.removeprefix("Could not parse LLM Output: ").removesuffix("`")
                #print(response)
            except Exception as e: 
                #response = "Currently, our database does not contain any resumes. We kindly request you to add a resume to our database before proceeding with any questions for the chatbot. Thank you for your cooperation and understanding."
                response = f"error: {e}"
                
        for chunk in response.split():
            full_response += chunk + " "
            sleep_time = random.triangular(0.0001, 0.8, 0.0001)
            time.sleep(sleep_time)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
                                
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
#If you have any questions, checkout our [documentation](add a link to our instruction manual here) 
