import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir,"db","chroma_db_metadata_r")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":5,"score_threshold":0.1}
)

while True:
    query=input("> ")

    relevant_docs = retriever.invoke(query)

    print("------Relevant--Docs-------")

    for i,doc in enumerate(relevant_docs, 1):
        print(f"\n-------Document-{i}-------\n{doc.page_content}\n********************")
        print(f"Source: {doc.metadata['source']}\n********************")
