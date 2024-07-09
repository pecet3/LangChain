import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(current_dir, "src")
persistent_directory = os.path.join(current_dir,"db","chroma_db_rdb2")

print(f"Book dir: {dir_path}")

if not os.path.exists(persistent_directory):
    print("Initializaing a vector store")

    if not os.path.exists(dir_path):
        raise FileNotFoundError(
            f"The directory {dir_path} does not exist..."
        )
    
    documents_list = []


    txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
    for file in txt_files:
    
        file_path = os.path.join(dir_path,file)
        print(file_path)
        loader = TextLoader(file_path, encoding='utf-8')
        txt_docs = loader.load()
        for doc in txt_docs:
            doc.metadata = {"source":file}
            documents_list.append(doc)

    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    docs = rec_char_splitter.split_documents(documents_list)

    
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


