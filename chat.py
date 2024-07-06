from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import dotenv_values

load_dotenv()

model=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.5, )

system_message = SystemMessage("You are a christian friendly bot")

chat_history = []

chat_history.append(system_message)

while True:
    query = input("Your Answer: ")

    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    lenHistory = len(chat_history)

    result = model.invoke(chat_history[lenHistory:lenHistory+1])



    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"Ai response: {response}") 