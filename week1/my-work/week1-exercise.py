import os
from dotenv import load_dotenv
from openai import OpenAI

# Load and verify environment variables
if not load_dotenv():
    raise RuntimeError("Failed to load .env file")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Define model configurations with their respective settings
MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "qwen3": {
        "url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())

def get_client_for_model(model_name):
    """
    Retrieves and returns a client object for interacting with the specified model.

    The function fetches configuration details corresponding to the given model
    name, validates the configuration, and creates an instance of the client
    appropriate for communicating with the model. Validation includes ensuring
    that the model has a valid API key and, if specified, a valid URL provided in
    the configuration.

    Parameters:
    model_name: str
        The name of the model for which the client is to be initialized.

    Raises:
    ValueError
        If the given model name does not exist in the configuration.
        If the API key is not configured for the specified model.
        If the URL provided in the configuration is invalid.

    Returns:
    OpenAI
        An instantiated OpenAI client object configured for the specified model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]

    if not config.get("api_key"):
        raise ValueError(f"API key not configured for model: {model_name}")

    if config.get("url") is None:
        return OpenAI(api_key=config["api_key"])
    else:
        if not isinstance(config["url"], str):
            raise ValueError(f"Invalid URL format for model: {model_name}")

        return OpenAI(base_url=config["url"], api_key=config["api_key"])

def ask_model(messages, model="gpt-4o-mini", stream=False):
    """
    Interact with the specified model to retrieve a chat completion.

    This function takes a list of message objects and a model name, then uses the
    OpenAI client to request a chat completion from the specified AI model. Optionally,
    the response can be streamed.

    Args:
        messages (List[dict]): A list of messages in dictionary format, where each
            dictionary contains keys such as "role" and "content".
        model (str): The name of the desired AI model to interact with. Defaults to
            "gpt-4o-mini".
        stream (bool): Indicates whether the response should be streamed. Defaults
            to False.

    Returns:
        Any: The response object returned by the OpenAI API client.
    """
    client = get_client_for_model(model)
    response = client.chat.completions.create(model=model, messages=messages, stream=stream)
    return response

def print_stream_response(response):
    """
    Prints the content of each chunk from a streamed response.

    The function iterates over a given response object, checking for the presence
    of data within the `chunk`. If valid content is available, it is printed to
    the standard output. This allows for handling chunked or streamed data in a
    progressive manner.

    Parameters:
    response: Iterable
        The response object to be iterated over. Each element of the response is
        expected to have a `choices` attribute with a nested structure containing
        a `delta.content`.

    Raises:
    Exception
        If an error occurs while processing the streamed response, the exception
        is caught and a message is printed to indicate the error.
    """
    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
    except Exception as e:
        print(f"\nError during streaming: {e}")

question = """
Explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

system_prompt = """
You're an expert software engineer and a great teacher, you're especially good with the Python programming language. 
Assume the user is a beginner and you're trying to teach them how to use Python.
"""

user_prompt = """
Answer the following question, giving an example to help explain:
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt + question}
]

response = ask_model(messages, model="qwen3", stream=True)
print_stream_response(response)