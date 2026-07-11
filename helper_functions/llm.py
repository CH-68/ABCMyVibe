import os
import tiktoken
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

if load_dotenv('.env'):
    # for local development
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']

client = OpenAI(api_key=OPENAI_KEY)


def get_completion(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_completion_stream(messages, model="gpt-4o-mini", temperature=0):
    """Returns a streaming generator for use with st.write_stream()."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def count_tokens(text, model="gpt-4o-mini"):
    """Estimate the number of tokens in a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
