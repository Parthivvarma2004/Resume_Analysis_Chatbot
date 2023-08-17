# Resume Analysis Chatbot
Welcome to the Resume Analysis Chatbot project! This chatbot is designed to assist users in analyzing and managing resumes efficiently. Below, you'll find an overview of the architecture, functionality, and tools utilized in this project.

You can access and test the chatbot at https://byte-busters-document-analysis-chatbot.streamlit.app/ 

## Architecture

### Frontend
The frontend of the chatbot is developed using Streamlit, a web application framework for Python. It provides an intuitive user interface for interacting with the chatbot and performing various tasks.

### Database
The chatbot communicates with a PostgreSQL database using the pgml (PostgresML) library. The database consists of two collections: one for storing original resumes and another for storing summarized versions of those resumes. This setup enhances retrieval and comparison efficiency. The database retains uploaded resumes even after users leave the webpage.

### Language Model
The core of the chatbot is powered by the GPT-3.5 Turbo language model from OpenAI. It handles tasks such as answering user queries, summarizing resumes, scheduling interviews, and ranking candidates.

### Tools and Toolkits
The chatbot leverages various tools and toolkits, including those provided by Langchain. For instance, the Zapier Toolkit facilitates interview scheduling.

## Functionality

### Upload Resumes
Users can upload multiple PDF resumes. The chatbot converts them to text, analyzes them, and generates a summarized version (250 tokens) using the language model. Both versions are transformed into embeddings and chunks for efficient storage in the database.

### Compare Resumes
The chatbot can compare up to 15 resumes and identify the most suitable candidates for different job positions. Users provide job requirements, and the chatbot ranks candidates based on their suitability. The Langchain ReAct agent is employed to reason and deduce the best candidates.

### Generate Spreadsheet
The chatbot generates spreadsheets containing essential candidate data like skills, experiences, projects, and degrees. This aids easy candidate comparison and evaluation. The generated spreadsheet can be converted to Excel format and downloaded.

### Schedule Interviews
Users can schedule interviews with selected candidates. The chatbot uses candidate names to extract email addresses, and then uses the GPT-3.5 Turbo model to create Google Calendar events and send email invitations via the Zapier Langchain agent.

### Answer Queries
Users can ask general questions related to resumes in the database. The chatbot employs the GPT-3.5 Turbo model and vector search algorithm to provide informative answers based on user queries.

### Clear Database
An option to clear all data from the database is provided for test users. This ensures a clean slate for app testing without interference from previous resumes.

## Keys and Authorization

### Open AI Key
Obtain an OpenAI key by creating an account and generating an API key through their API plan.

### Zapier NLA Key
Zapier provides a server-side implementation. The current email for sending interview invites is hiringmanagertesthypeai@gmail.com. You can replace this email by creating a Zapier account and generating your personal NLA key.

### Database
The project uses a free 5 GB PostgreSQL database from PostgresML. You can switch to a larger database by modifying the database URL. Multiple users can be accommodated by creating separate collections for each user.

Feel free to contribute, enhance, and customize the chatbot for your needs!
