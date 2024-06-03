!pip install langchain langchain-experimental langchain-community langchain-openai openai chromadb pypdf sentence_transformers gradio langchain-together

#####################

import os

#document loader
from langchain_community.document_loaders import PyPDFLoader

# vector store
from langchain_community.vectorstores import Chroma

#llm
from langchain_openai import OpenAI

#####################

loader = PyPDFLoader("/content/aws-overview.pdf") #upload the document you wish to use for chatbot
#https://d1.awsstatic.com/whitepapers/aws-overview.pdf
pages = loader.load()

#####################

len(pages)

#####################

pages[16]

#####################

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=500, chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  docs = text_splitter.split_documents(documents)
  return docs

#####################

new_pages = split_docs(pages)
len(new_pages)

#####################

new_pages[500].page_content

#####################

new_pages[499].page_content

#####################

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(new_pages, embedding_function)

#####################

from langchain_together import Together


llm = Together(
    model="meta-llama/Llama-2-70b-chat-hf",
    max_tokens=256,
    temperature=0,
    top_k=1,
    together_api_key="ENTER YOUR API KEY"

    #https://api.together.ai/settings/api-keys
)

#####################

retriever = db.as_retriever(similarity_score_threshold = 0.9)

#####################

from langchain.prompts import PromptTemplate
prompt_template = """Please answer questions related AWS (Amazon web services). Try explaining in simple words. Answer in less than 100 words. If you don't know the answer simply respond as "Don't know man!"
 CONTEXT: {context}
 QUESTION: {question}"""

PROMPT = PromptTemplate(template = f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

#####################

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type='stuff',
    retriever= retriever,
    input_key = 'query',
    return_source_documents = True,
    chain_type_kwargs={"prompt":PROMPT},
    verbose=True

)

#####################

query = input()
response = chain(query)
response['result']

#####################
