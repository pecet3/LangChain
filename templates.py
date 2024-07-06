from  langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo")

template = "Tell me a joke about {topic}"

prompt_template = ChatPromptTemplate(template)

prompt = prompt_template.invoke({"topic":"cats"})
print(prompt)