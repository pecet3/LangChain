from  langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")


prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are an expert product reviewer. You are responding in polish"),
    ("human", "List the main features of product - {product_name}")
])


def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a product reviewer"),
            ("human", "Given these features:{features}, list the pros of these features")
        ]
    )

    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a product reviewer"),
            ("human", "Given these features:{features}, list the cons of these features")
        ]
        )

    return pros_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\nCons:\n{cons}"

product = input("Product: ")

cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x))| model | StrOutputParser()
)
pros_branch = (
    RunnableLambda(lambda x: analyze_pros(x)) | model |StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch, "cons":cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

try:
    result = chain.invoke({"product_name": product})
    print(result)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Traceback:")
