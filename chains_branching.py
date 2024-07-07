from  langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

postivie_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service bot"),
        ("human", "Generate a thank you for a positive feedback: {feedback}")
    ]
)

negative_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service bot"),
        ("human", "Generate a thank you for a negative feedback: {feedback}")
    ]
)

neutral_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service bot"),
        ("human", "Generate a thank you for a neutral feedback: {feedback}. Ask for provide more details.")
    ]
)

escalate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service bot"),
        ("human", "Generate a thank you for a escalate feedback: {feedback}")
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service bot"),
        ("human", "Classify the feedback as postivie, negative, neutral or escalate: {feedback}")
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        postivie_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_template | model | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x,
        neutral_template | model | StrOutputParser()
    ),
    escalate_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches
review = input("Review: ")
result = chain.invoke({"feedback":review})
print(result)