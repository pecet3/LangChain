from  langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# template = "Tell me a joke about {topic}"

# prompt_template = ChatPromptTemplate.from_template(template)
# prompt = prompt_template.invoke({"topic":topic})



messages = [
    ("system", "You are a christian bot who tells jokes about {topic}. You are responding in polish"),
    ("human", "Tell me {joke_count} jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

topic = input("Topic: ")
count = input("Count: ")


prompt = prompt_template.invoke({"topic": topic,"joke_count":count})

result = llm.invoke(prompt)
print(result.content)