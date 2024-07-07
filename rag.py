import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books","socket.txt")
persistent_directory = os.path.join(current_dir,"db","chroma_db")

if not os.path.exists(persistent_directory):
    print("Initializaing a vector store")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist..."
        )
    
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"Sample Chunk: {docs[0].page_content}")

    embedings= OpenAIEmbeddings(
        model="text-embeding-3-small"
    )

    print("Finished embedings")

    db = Chroma.from_documents(
        docs, embedings, persist_directory=persistent_directory
    )

    print("Vector store created")

else:
    print("Vector store exists")
