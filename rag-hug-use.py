import os
from  langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    ("system", "You are a computer science bot specialized in c language"),
    ("human", "Based on these fragments: {documents} answer to a question: {query}")
]



current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir,"db","chroma_db_metadata_hug_2000")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,"score_threshold":0.1}
)

# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k":3}
# )

while True:
    query=input("> ")
    prompt_template = ChatPromptTemplate.from_messages(messages)

    relevant_docs = retriever.invoke(query)
  
    print("------Relevant--Docs-------")
    documents = []
    for i,doc in enumerate(relevant_docs, 1):
        print(f"\n\n-------Document-{i}-------\n\n{doc.page_content}\n********************")
        print(f"Source: {doc.metadata['source']}\n********************")
        documents.append(f"Fragment no: {i}\n"+doc.page_content)

    chain = prompt_template | model | StrOutputParser()   
    result = chain.invoke({"documents":documents,"query":query})
    print(f"\n {result}")
    



