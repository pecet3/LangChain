import os
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


print("Loading...")

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir,"db","chroma_db_rdb3")


model = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
    )

create_stuff_documents_chain

contextualize_q_system_prompt = """

"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_q_prompt)

qa_system_prompt = """
You are an assistant working in Raiffeisen Digital Bank for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
If the question is not related with Raiffeisen ask that you don't know the answer.
Assume user always ask question related to Raiffeisen.
{context}
"""

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model,qa_prompt)

rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

def run_chat():
    chat_history=[]
    while True:
        query = input("> ")
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"\n[AI]\n{result["answer"]}\n")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

run_chat()