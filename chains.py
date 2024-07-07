from  langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    ("system", "You are a christian bot who tells jokes about {topic}. You are responding in polish"),
    ("human", "Tell me {joke_count} jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

topic = input("Topic: ")
count = input("Count: ")

chain = prompt_template | model |StrOutputParser()

result = chain.invoke({"topic": topic,"joke_count":count})
print(result)