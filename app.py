import os
import chainlit as cl
from chainlit import AskUserMessage, Message, on_chat_start
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import openai
import base64
from langsmith import traceable
from prompt import SYSTEM_PROMPT
import json

index = None

api_key = os.getenv("OPENAI_API_KEY")

endpoint_url = "https://api.openai.com/v1"

client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.3,
    "max_tokens": 1000
}


@on_chat_start
async def on_chat_start():
    documents = SimpleDirectoryReader("data").load_data()
    global index
    index = VectorStoreIndex.from_documents(documents)
    


@traceable
@cl.on_message
async def on_message(message: cl.Message):
    retrieved_docs = index.retrieve(message.content)
    # Combine retrieved documents into a single context
    context = " ".join([doc['text'] for doc in retrieved_docs])
    

    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    message_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    message_history.append({"role": "system", "content": context})
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()
    
    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history, 
                                                  stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
        