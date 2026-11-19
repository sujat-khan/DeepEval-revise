import os
from dotenv import load_dotenv
from deepeval.tracing import observe, update_current_trace
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


@observe(type="agent")
def my_agent(query: str) -> str:
    response = llm.invoke(query)
    answer = str(response.content)
    update_current_trace(input=query, output=answer)
    return answer
