import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
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
        color: black;
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
        color: #14213d;
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
    .content-box {
        background-color: rgba(255, 255, 255, 0.8); /* White with transparency */
        border-radius: 10px; 
        padding: 20px; 
        margin: 20px auto;
        color: black;
        width: 80%; /* Adjust width as needed */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Optional shadow for depth */
        text-align: center; /* Center-align the content */
    }
    .mantra {
        font-size: 30px;
        font-weight: bold;
        color: orange;
        animation: fadeIn 2s ease-in-out;
        background-color: rgba(255, 255, 255, 0.8); /* White with transparency */
        border-radius: 10px; 
        padding: 20px; 
        margin: 20px auto; 
        width: 80%; /* Adjust width as needed */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Optional shadow for depth */
        text-align: center; /* Center-align the content */
    }
    .mantra-meaning {
        font-size: 20px;
        color: black;
    }
    h1{
        color: orange;
    }
    
    h2{
        color: orange;
    }
    
    h3{
        color: orange;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* Sidebar header */
    .css-1v3fvcr {
        color: white !important;
    }

    /* Sidebar text */
    .css-10trblm {
        color: white !important;
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
<div class='content-box' id='home'>
    <h2>Welcome to Geeta GPT</h2>
    <p>This application provides advice and answers based on the teachings of the Bhagavad Gita. Simply enter your question to get started.</p>
    <p class='mantra'>ॐ कृष्णाय वासुदेवाय हरये परमात्मने॥<br>प्रणत: क्लेशनाशाय गोविंदाय नमो नम:॥</p>
    <p class='mantra-meaning'>English Translation: "Om Krishnaya Vasudevaya Haraye Paramatmane, Pranatah Kleshanashaya Govindaya Namo Namah".<br>Meaning: This mantra is a salutation to Lord Krishna, the Supreme Soul, who removes the sufferings of the devotees who surrender to Him.</p>
</div>
"""
about_content = """
<div class='content-box' id='about'>
    <h2>About Geeta GPT</h2>
    <p>Geeta GPT is powered by advanced AI technology, utilizing the wisdom of the Bhagavad Gita to offer guidance and insights. Our goal is to make the ancient teachings accessible to everyone.</p>
</div>
"""
contact_us_content = """
<div class='content-box' id='contact-us'>
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
    # Geeta GPT functionality

    # User input for question with placeholder
    user_question = st.text_input("", "", placeholder="Enter your question")

    # Submit button with custom style
    if st.button("Submit", key="submit", help="Click to submit your question"):
        if user_question:
            # Initialize the ChatGroq model
            try:
                mistral_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
            except Exception as e:
                st.error("Error initializing ChatGroq model.")
                mistral_llm = None

            # Read the CSV file
            csv_file_path = 'modified_meaning.csv'
            try:
                df = pd.read_csv(csv_file_path, nrows=600)
            except Exception as e:
                st.error("Error loading CSV file.")
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

                # Embed and store in FAISS
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                try:
                    faiss_index = FAISS.from_documents(chunked_documents, embedding_model)
                except Exception as e:
                    st.error("Error creating FAISS index.")
                    faiss_index = None

                if faiss_index is not None:
                    # Generate prompt template
                    template = """
                    You are Geeta-GPT, an AI assistant designed to help with questions related to the Bhagavad Gita. 
                    Given the following passages and a user's question, provide a comprehensive response based on the teachings of the Bhagavad Gita.

                    Passages:
                    {context}

                    User Question:
                    {question}

                    Response:
                    """
                    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

                    # Create QA chain
                    qa_chain = LLMChain(
                        llm=mistral_llm,
                        prompt=prompt_template
                    )

                    # Find relevant documents
                    docs = faiss_index.similarity_search(user_question, k=3)
                    docs_page_content = [d.page_content for d in docs]
                    context = "\n\n".join(docs_page_content)

                    # Generate answer using the QA chain
                    try:
                        answer = qa_chain.run({"context": context, "question": user_question})
                        # Display answer with a custom response box
                        st.markdown(
                            f"<div class='response-box'><p class='response-text'>{answer}</p></div>",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error("Error generating response.")
        else:
            st.warning("Please enter a question.")
