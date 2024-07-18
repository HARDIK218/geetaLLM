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
st.set_page_config(page_title="Geeta GPT", page_icon="🕉️", layout="wide")

# Sidebar configuration
st.sidebar.title("Geeta GPT")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
navigation = st.sidebar.radio("Go to", ["Geeta-GPT", "Home", "About", "Contact Us"])

# Custom CSS to add background image and other styles
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://blog.cdn.level.game/2024/05/bhagavad-gita-6-chapter--meditation-1.webp') no-repeat center center fixed; 
        background-size: cover; 
        height: 100vh; 
        overflow: auto;  
    }
    .title-container {
        background-color: rgba(255, 255, 255, 0.8); /* White with transparency */
        border-radius: 10px; 
        padding: 20px; 
        margin: auto; 
        width: 60%; /* Adjust width as needed */
        text-align: center; 
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Optional shadow for depth */
    }
    input::placeholder {
        color: #f28f2c;
        opacity: 1; 
    }
    .response-box {
        border: 2px solid #4B8BBE; 
        border-radius: 10px; 
        padding: 10px; 
        background-color: rgba(249, 249, 249, 0.8);
    }
    .response-text {
        color: #FFFFFF;
    }
    .submit-btn {
        background-color: #4B8BBE; 
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
    }
    .submit-btn:hover {
        background-color: #3a8cbf;
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the title and subtitle in a box only if navigation is "Geeta-GPT"
if navigation == "Geeta-GPT":
    st.markdown(
        "<div class='title-container'>"
        "<h1 class='title'>Geeta GPT</h1>"
        "<h3 class='subtitle'>Ask a question and get advice based on Bhagavad Geeta</h3>"
        "</div>",
        unsafe_allow_html=True
    )

# Content for Home, About, and Contact Us sections
home_content = """
<div id='home'>
    <h2>Welcome to Geeta GPT</h2>
    <p>This application provides advice and answers based on the teachings of the Bhagavad Gita. Simply enter your question to get started.</p>
</div>
"""
about_content = """
<div id='about'>
    <h2>About Geeta GPT</h2>
    <p>Geeta GPT is powered by advanced AI technology, utilizing the wisdom of the Bhagavad Gita to offer guidance and insights. Our goal is to make the ancient teachings accessible to everyone.</p>
</div>
"""
contact_us_content = """
<div id='contact-us'>
    <h2>Contact Us</h2>
    <p>If you have any questions or feedback, please reach out to us at <a href="mailto:support@geetagpt.com">support@geetagpt.com</a>.</p>
</div>
"""

# Display content based on navigation selection
if navigation == "Home":
    st.markdown(home_content, unsafe_allow_html=True)
elif navigation == "About":
    st.markdown(about_content, unsafe_allow_html=True)
elif navigation == "Contact Us":
    st.markdown(contact_us_content, unsafe_allow_html=True)
else:
    # User input for question
    user_question = st.text_input("Enter your question:", "")

    # Initialize the ChatGroq model
    try:
        mistral_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
    except Exception as e:
        st.error(f"Error initializing ChatGroq model.")
        mistral_llm = None

    # Read the CSV file
    csv_file_path = 'modified_meaning.csv'
    try:
        df = pd.read_csv(csv_file_path, nrows=600)
    except Exception as e:
        st.error(f"Error loading CSV file.")
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
        except Exception as e:
            st.error(f"Error initializing FAISS database.")
            retriever = None

        # Create prompt template
        prompt_template = """
        Note: While returning the final answer, please print a little bit of context from docs that you have used to generate the answer.
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

                # Display the response in a box
                st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                st.markdown(f"<p class='response-text'>{formatted_text}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing question.")
        elif not user_question:
            st.write("Please enter a question to get advice based on Bhagavad Geeta.")
        else:
            st.write("Please ensure the model and retriever are initialized correctly.")
    else:
        st.write("Please ensure the CSV file is loaded correctly.")
