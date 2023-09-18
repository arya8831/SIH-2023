from langchain import HuggingFaceHub, PromptTemplate,  LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.schema import retriever
from langchain.vectorstores import Chroma
import textwrap
import gradio as gr
import os

# HugginfFAce API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#pdf_loader = DirectoryLoader('data', glob="**/*.pdf")
txt_loader = DirectoryLoader('data', glob="**/*.txt")

#take all the loader
loaders = [txt_loader]

#lets create document
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=40)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vectorStore = Chroma.from_documents(documents, embeddings)

repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 5000}
)

def askanything(question, history):
  template = """
  You are an intelligent chatbot. Help the following question with brilliant answers.
  Question: {question}
  Answer:"""
  prompt = PromptTemplate(template=template, input_variables=["question"])

  llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

  
  response = llm_chain.run(question)

  wrapped_text = textwrap.fill(
      response, width=100, break_long_words=False, replace_whitespace=False)
  return wrapped_text

def chatwithtext(query, history):
  chain = RetrievalQA.from_chain_type(llm=falcon_llm, chain_type="stuff", retriever=vectorStore.as_retriever())

  
  response = chain.run(query)
  #wrapped_text = textwrap.fill(
  #    response, width=100, break_long_words=False, replace_whitespace=False)
  return response

with gr.Blocks() as demo:
    gr.Markdown("Ask Questions to Falcon.")
    with gr.Tab("General Question"):
        gr.ChatInterface(askanything)
    with gr.Tab("Questions related to rules and regulations"):
        gr.ChatInterface(chatwithtext)


# create the link
demo.launch()

