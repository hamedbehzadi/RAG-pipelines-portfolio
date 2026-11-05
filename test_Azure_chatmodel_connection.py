import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    max_tokens=300,
    timeout=60,
)

response = llm.invoke("Explain RAG in two sentences.")
print(response.content)