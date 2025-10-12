import os
from langchain_openai import ChatOpenAI


api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1.6-250615")


response = llm.invoke("你好，介绍一下你自己")
print(response.content)

