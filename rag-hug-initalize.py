import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(current_dir, "books")
persistent_directory = os.path.join(current_dir,"db","chroma_db_metadata_hug")

print(f"Book dir: {dir_path}")

if not os.path.exists(persistent_directory):
    print("Initializaing a vector store")

    if not os.path.exists(dir_path):
        raise FileNotFoundError(
            f"The directory {dir_path} does not exist..."
        )
    
    book_files = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
    documents_list = []
    for book_file in book_files:
        file_path = os.path.join(dir_path,book_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source":book_file}
            documents_list.append(doc)

    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)

    docs = rec_char_splitter.split_documents(documents_list)

    print(f"Sample Chunk: {docs[0].page_content}")
    
    print("Initalize embedings")
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Finished embedings")

    print("Initalize chroma generate process")

    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("Finished chroma generate process")

    print("Vector store created")
else:
    print("Vector store exists")


