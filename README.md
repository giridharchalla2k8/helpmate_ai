# **Build a RAG System**

## **Problem Statement:**

Organizations stores the manuals, policy documents, legal contracts or any other documents in unstructured format like pdfs. In this case study use any Life Insurance Policy pdf document like " Principal-Sample-Life-Insurance-Policy.pdf", and retrieve instant and accurate answers from the document.
Traditional keyword-based search leads to inefficient contextual information, missed insights and further leads to frustrated users.
The case study should concentrate on the below points. 
 	Understand the context of the question,
 	Match semantically similar but differently worded content,
 	And synthesize answers across multiple sections.

## Packages/Libraries to be installed/imported

### Packages

!pip install -U -q llama-index openai llama-index-core llama-index-embeddings-openai

!pip install llama-index-llms-openai

!pip install pymupdf

!pip install -U -q langchain pdfplumber langchain-community sentence-transformers google-colab pypdf opentelemetry-sdk

!pip install chromadb

!pip install -U -q langchain openai tiktoken langchain-openai

!pip install llama-index-llms-langchain

### Importing libraries

from llama_index.llms.openai import OpenAI

from llama_index.core.llms import ChatMessage

import os

import openai

from dotenv import load_dotenv

import nest_asyncio

nest_asyncio.apply()

from google.colab import drive

drive.mount('/content/drive')

from pathlib import Path

from llama_index.core import download_loader

from llama_index.readers.file import PyMuPDFReader

from collections import Counter

from llama_index.core.node_parser import SimpleNodeParser

from llama_index.core import VectorStoreIndex

from IPython.display import display, HTML

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.openai import OpenAI

from llama_index.core import Settings

from llama_index.core.service_context import ServiceContext

from llama_index.core.node_parser import TokenTextSplitter


from langchain.vectorstores import Chroma,FAISS

from langchain.document_loaders import PyPDFDirectoryLoader # Using standard PDF loader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema.runnable import RunnablePassthrough

from langchain.schema.output_parser import StrOutputParser

from langchain.prompts import ChatPromptTemplate

from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors import CrossEncoderReranker

from sentence_transformers.cross_encoder import CrossEncoder # For reranker

from langchain_community.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings

#from langchain.text_splitter import CharacterTextSplitter

from langchain.document_loaders import PyPDFLoader

from langchain.chains import create_history_aware_retriever,create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.vectorstores import Chroma,FAISS

import chromadb


## **Results:**

Llamaindex query results:
## Query 1
Query("What risks or events are covered under this policy?")
Query :  What risks or events are covered under this policy?
Response :  The policy covers accidental death and dismemberment resulting from events such as willful self-injury, disease or medical treatment complications, participation in certain criminal activities, involvement in specific aeronautic activities, military duty, war, alcohol use exceeding legal limits, operation of a vehicle or boat under the influence, unauthorized drug use, and injuries not related to employment for wage or profit.
- Page 58: This policy has been updated effective  January 1, 2014        PART IV - BENEFITS  GC 6015   Section B - Member Accidental Death and  Dismemberment Insurance, Page 6      a.  willful self-injury or se...
- Page 27: This policy has been updated effective  January 1, 2014  PART III - INDIVIDUAL REQUIREMENTS AND RIGHTS  GC 6006  Section A - Eligibility, Page 2    If a Member's Dependent is employed and is covered u...
- Page 13: This policy has been updated effective  January 1, 2014  GC 6002   PART I - DEFINITIONS, PAGE 5      a.  A licensed Doctor of Medicine (M.D.) or Osteopathy (D.O.); or    b.  any other licensed health ...
## Query 2
Query("what is the life insurance coverage for disability")
Query("what is the life insurance coverage for disability")
The life insurance coverage for disability includes provisions for ADL Disability or Total Disability. To be eligible for Coverage During Disability, a Member must meet certain qualifications such as becoming ADL Disabled or Totally Disabled while insured for Member Life Insurance, being under the care of a Physician, providing proof of disability when required, and undergoing Medical Examinations or Evaluations as needed. Written proof of ADL Disability or Total Disability must be submitted to the insurance provider within a specified timeframe, and further proof may be requested periodically. If the Member passes away while disabled, final proof of disability continuation must be provided to the insurance provider.
- Page 49: This policy has been updated effective  January 1, 2014  PART IV - BENEFITS  GC 6013   Section A - Member Life Insurance, Page 4      Payment of benefits will be subject to the Beneficiary and Facilit...
- Page 51: This policy has been updated effective  January 1, 2014  PART IV - BENEFITS  GC 6013   Section A - Member Life Insurance, Page 6    Coverage During Disability will cease on the earliest of:    (1) the...
- Page 50: This policy has been updated effective  January 1, 2014  PART IV - BENEFITS  GC 6013   Section A - Member Life Insurance, Page 5    The Principal may require that a ADL Disabled or Totally Disabled Me...
## Query 3
Query("Summarize the key benefits from the insurance policy documents.")
Query :  Summarize the key benefits from the insurance policy documents.
Response :  The risks or events covered under this policy include accidental death and dismemberment that are not a result of willful self-injury, disease or medical treatment complications, participation in criminal activities, certain aeronautic activities, military duty, war, excessive alcohol consumption, drug use, or injuries sustained during employment for wage or profit.
- Page 58: This policy has been updated effective  January 1, 2014        PART IV - BENEFITS  GC 6015   Section B - Member Accidental Death and  Dismemberment Insurance, Page 6      a.  willful self-injury or se...
- Page 27: This policy has been updated effective  January 1, 2014  PART III - INDIVIDUAL REQUIREMENTS AND RIGHTS  GC 6006  Section A - Eligibility, Page 2    If a Member's Dependent is employed and is covered u...
- Page 13: This policy has been updated effective  January 1, 2014  GC 6002   PART I - DEFINITIONS, PAGE 5      a.  A licensed Doctor of Medicine (M.D.) or Osteopathy (D.O.); or    b.  any other licensed health ...
## Query 4
Query("What riders or add-ons are available?")
Query :  What riders or add-ons are available?
Response :  The risks or events covered under this policy include accidental death and dismemberment that are not a result of willful self-injury, disease or medical treatments, participation in criminal activities, certain aeronautic activities, military duty, war, excessive alcohol consumption, drug use, and injuries sustained during employment for wage or profit.
- Page 58: This policy has been updated effective  January 1, 2014        PART IV - BENEFITS  GC 6015   Section B - Member Accidental Death and  Dismemberment Insurance, Page 6      a.  willful self-injury or se...
- Page 27: This policy has been updated effective  January 1, 2014  PART III - INDIVIDUAL REQUIREMENTS AND RIGHTS  GC 6006  Section A - Eligibility, Page 2    If a Member's Dependent is employed and is covered u...
- Page 13: This policy has been updated effective  January 1, 2014  GC 6002   PART I - DEFINITIONS, PAGE 5      a.  A licensed Doctor of Medicine (M.D.) or Osteopathy (D.O.); or    b.  any other licensed health ...



## **Langchain results: (Screenshots)**
 
In the attachment.
 
