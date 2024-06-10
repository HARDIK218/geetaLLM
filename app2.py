import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
import pandas as pd
from bs4 import BeautifulSoup
from html2text import HTML2Text
import os
import re
from langchain_groq import ChatGroq

def format_response(response: str) -> str:
    entries = re.split(r" (?<=]), (?=\[)", response)
    return [entry.strip("[]") for entry in entries]

# Read CSV and process data
csv_file_path = 'modified_meaning.csv'
column_name = 'meaning'
df = pd.read_csv(csv_file_path, nrows=600)

docs_transformed = []

# Create an HTML2Text object
html2text = HTML2Text()

# Extract data from the specified column into a list
for index, row in df.iterrows():
    html_content = row[column_name]
    html_content = str(html_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = html2text.handle(str(soup))
    docs_transformed.append(plain_text)

class PageContentWrapper:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

# Wrap the transformed documents
docs_transformed_wrapped = [PageContentWrapper(content) for content in docs_transformed]

# Use CharacterTextSplitter to chunk the documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(docs_transformed_wrapped)

# Create FAISS vector store
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever()

def get_mistral_response(question, retriever):
    os.environ["GROQ_API_KEY"] = "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"

    mistral_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")

    prompt_template = """
    note while returning final answer please print little bit of context from docs that you have used to generate answer
    ### [INST] Instruction: Answer the question based on your docs knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | llm_chain
    )

    result = rag_chain.invoke({'question': question})

    text = result['text']
    formatted_text = text.replace('\n', ' ').replace('. ', '.\n\n')

    return formatted_text

st.title("Geeta GPT")
st.text("answer to your questions")
jd = st.text_area("question")
submit = st.button("Submit")

if submit:
    if jd:
        response = get_mistral_response(jd, retriever)
        st.subheader(response)
