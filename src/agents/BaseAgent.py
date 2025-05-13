import os
import json
import logging
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from typing import Dict, Any, Callable, Literal, List
from langchain_core.language_models.base import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_together import Together
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

import tiktoken

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

def run_with_retry(func: Callable, max_retries: int = 5, min_wait: float = 30.0, max_wait: float = 90.0, arg=None, **kwargs):
    """
    Execute a function with exponential backoff and jitter using tenacity.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        arg: Single positional argument to pass to the function (if needed)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function if successful
        
    Raises:
        Exception: If all retries fail
    """
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def wrapped_func():
        try:
            if arg is not None:
                return func(arg)
            else:
                return func(**kwargs)
        except Exception as e:
            logging.warning(f"Retry triggered: {func.__name__} failed with error: {str(e)}")
            raise
        
    return wrapped_func()

def cut_off_tokens(text: str, max_tokens: int, encoding_name: str = "gpt-4o"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        # cut off the last max_tokens tokens
        return encoding.decode(tokens[-max_tokens:])
    return text

class BaseAgent():
    
    def __init__(
        self,
        api_type: Literal["azure"],
        api_key: str,
        model_name: Literal["gpt-4o", "gpt-4o-mini", "o3-mini"] = None,
        endpoint: str=None,
        max_completion_tokens=5000,
        **kwargs
    ):  
        # get endpoint using model type
        self.endpoint = endpoint
        self.api_key = api_key

        # load model config
        self.model_name = model_name
        
        self.api_type = api_type
        
        self.max_completion_tokens = max_completion_tokens
        
        # get the model            
        self.llm = self.get_model(
            api=self.api_type,
            model_name=self.model_name,
            api_key=self.api_key,
            endpoint=self.endpoint,
            **kwargs
        )

    def get_model(
            self,
            api: str,
            api_key: str,
            model_name: str,
            endpoint: str = None,
            **kwargs
    ) -> BaseLanguageModel:
        """
        Get the appropriate language model based on the API type
        
        Args:
            api: The API provider ('together', 'anthropic', 'openai', 'google')
            api_key: The API key for the provider
            model: The model name
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            A language model instance
        """
        if (model_name not in ["o3-mini", "o3-preview"]):
            # remove max_completion_tokens from kwargs since it's not supported
            # by all models
            if "max_completion_tokens" in kwargs:
                del kwargs["max_completion_tokens"]
        
        llm = None
        if (api == "together"):
            llm = Together(
                model=model_name,
                together_api_key=api_key,
                **kwargs
            )
        elif (api == "anthropic"):
            llm = ChatAnthropic(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        elif (api == "openai"):
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        elif (api == "google"): 
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                **kwargs
            )
        elif (api == "azure"):
            llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=model_name,
                api_key=api_key,
                api_version="2024-12-01-preview",
                **kwargs
            )
        else:
            raise ValueError(f"Invalid API: {api}")
        return llm

    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Base method for generating code.
        
        Args:
            input_query: The user query to process
            **kwargs: Additional arguments to pass to the agent graph
            
        Returns:
            Dict[str, Any]: The result from the agent graph or an error dict
        """
        
        assert self.agent_graph is not None, "Agent graph is not set"
        
        # Extract input_query from kwargs
        input_query = kwargs.pop("input_query", None)
        if input_query is None:
            return {"error": "input_query is required"}
        
        try:
            # Prepare inputs for agent graph
            inputs = {
                "messages": [("user", input_query)],
                **kwargs  # Pass remaining kwargs to the agent graph
            }
            
            # Invoke the agent graph and return the result
            result = self.agent_graph.invoke(inputs)
            return result
            
        except Exception as e:
            print(f"Error generating code: {e}")
            raise e
