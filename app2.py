import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
import pandas as pd
from bs4 import BeautifulSoup

# Set up API key for ChatGroq
os.environ["GROQ_API_KEY"] = "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"

# Streamlit app title and description
st.title("Geeta GPT")
st.write("Ask a question and get advice based on Bhagwad Geeta")

# User input for question
user_question = st.text_input("Enter your question:")

# Initialize the ChatGroq model
try:
    mistral_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
    st.write("ChatGroq model initialized successfully.")
except Exception as e:
    st.error(f"Error initializing ChatGroq model: {e}")
    mistral_llm = None

# Read the CSV file
csv_file_path = 'modified_meaning.csv'
try:
    df = pd.read_csv(csv_file_path, nrows=600)
    st.write("CSV file loaded successfully.")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    df = None

if df is not None:
    column_name = 'meaning'

    # Transform content from the specified column
    docs_transformed = []

    for index, row in df.iterrows():
        html_content = row[column_name]
        html_content = str(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(separator="\n")
        docs_transformed.append(plain_text)

    class PageContentWrapper:
        def __init__(self, page_content, metadata={}):
            self.page_content = page_content
            self.metadata = metadata

    # Wrap and chunk documents
    docs_transformed_wrapped = [PageContentWrapper(content) for content in docs_transformed]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(docs_transformed_wrapped)

    # Initialize FAISS database
    try:
        db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        retriever = db.as_retriever()
        st.write("FAISS database initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing FAISS database: {e}")
        retriever = None

    # Create prompt template
    prompt_template = """
    note while returning final answer please print little bit of context from docs that you have used to generate answer
    ### [INST] Instruction: Answer the question based on your docs knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {user_question} [/INST]
    """

    prompt = PromptTemplate(input_variables=["context", "user_question"], template=prompt_template)

    if mistral_llm is not None:
        llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    else:
        llm_chain = None

    if retriever is not None and llm_chain is not None:
        rag_chain = ({"context": retriever, "user_question": RunnablePassthrough()} | llm_chain)
    else:
        rag_chain = None

    if user_question and rag_chain:
        try:
            result = rag_chain.invoke(user_question)
            text = result['text']

            # Format the response text
            formatted_text = text.replace('\n', ' ').replace('. ', '.\n\n')

            # Display the response
            st.write(formatted_text)
        except Exception as e:
            st.error(f"Error processing question: {e}")
    elif not user_question:
        st.write("Please enter a question to get advice based on Bhagwad Geeta.")
    else:
        st.write("Please initialize the model and retriever correctly.")
else:
    st.write("Please ensure the CSV file is loaded correctly.")
