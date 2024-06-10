import warnings
import streamlit as st
warnings.filterwarnings('ignore')

from langchain.text_splitter import CharacterTextSplitter
#from langchain.document_transformers import Html2TextTransformer
#from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
#from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain


import os
import re
#import requests
#from google.colab import userdata
#from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def format_response (response: str) -> str:
    entries = re.split(r" (?<=]), (?=\[)", response)
    return [entry.strip("[]") for entry in entries]


os.environ["GROQ_API_KEY"] = "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"

mistral_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")

import pandas as pd
from bs4 import BeautifulSoup
from html2text import HTML2Text

# Replace 'your_file.csv' with the actual path to your CSV file
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


# Assuming plain_text is the content you want to chunk
docs_transformed_wrapped = [PageContentWrapper(content) for content in docs_transformed]

# Now use docs_transformed_wrapped with CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(docs_transformed_wrapped)


db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever()


# Create prompt template
prompt_template = """
note while returning final answer please print little bit of context from docs that you have used to generate answer
### [INST] Instruction: Answer the question based on your docs knowledge. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)


rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
#result = rag_chain.invoke('jd')

result = rag_chain.invoke("I have recently changed my office team and now i am not felling inclined what should i do")
text = result['text']

# Add line breaks to format the text as a paragraph
formatted_text = text.replace('\n', ' ')  # Replace existing line breaks with spaces
formatted_text = formatted_text.replace('. ', '.\n\n')  # Add double line breaks after periods

# Print the formatted text
#print(formatted_text)

print(formatted_text)

