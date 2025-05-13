import httpx
import os
from openai import OpenAI
from openai import AzureOpenAI
import tenacity
import json

from ..llm import AZURE_OPENAI_MODEL_NAME_MAP

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=5), stop=tenacity.stop_after_attempt(10), reraise=True)
def api_call_single(client: OpenAI, model: str, messages: list[dict], temperature: float = 0.0, **kwargs):
    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # Ensure messages is a list
        temperature=temperature,
        **kwargs
    )
    return response

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=5), stop=tenacity.stop_after_attempt(10), reraise=True)
def api_function_call_single(client: OpenAI, model: str, messages: list[dict], tools: list[dict], tool_choice:dict, temperature: float = 0.0, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        tool_choice=tool_choice,
        **kwargs
    )
    return response

def call_azure_openai(llm: str, messages: list[dict], temperature: float = 0.0, **kwargs):
    """
    Call the OpenAI API asynchronously to a list of messages using high-level asyncio APIs.
    """
    client = AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=100
        )
        )
    )
    model = AZURE_OPENAI_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unsupported LLM model: {llm}")
    response = api_call_single(client, model, messages, temperature, **kwargs)
    return response
    
def function_call_azure_openai(llm: str, messages: list[dict], tools: list[dict], tool_choice: dict=None, temperature: float = 0.0, **kwargs):
    """
    Call the OpenAI API asynchronously to a list of messages using high-level asyncio APIs.
    """
    client = AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=100
        )
        )
    )
    model = AZURE_OPENAI_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unsupported LLM model: {llm}")
    response = api_function_call_single(client, model, messages, tools, tool_choice, temperature, **kwargs)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    outputs = {}
    if tool_calls:
        outputs = json.loads(tool_calls[0].function.arguments)
    return outputs