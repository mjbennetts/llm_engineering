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
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "groq": {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY")
    },
    "ollama": {
        "model": "qwen3:14b",
        "url": "http://localhost:11434/v1",
        "api_key": "ollama"
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())

def get_client_for_model(service):
    """
    Retrieves a client instance for the specified model service.

    The function validates if the given service exists in the MODEL_CONFIGS
    dictionary and has the necessary configuration such as an API key and, if
    required, a valid URL. Depending on the presence of a URL in the configuration,
    an OpenAI client is instantiated either with a base URL or without it.

    Parameters:
    service : str
        The name of the model service to retrieve the client for.

    Raises:
    ValueError
        If the service is not found in MODEL_CONFIGS, its API key is missing,
        or the URL (if provided) is invalid.

    Returns:
    OpenAI
        An instance of the OpenAI client configured for the specified model.
    """
    if service not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {service}")

    config = MODEL_CONFIGS[service]

    if not config.get("api_key"):
        raise ValueError(f"API key not configured for model: {service}")

    if config.get("url") is None:
        return OpenAI(api_key=config["api_key"])
    else:
        if not isinstance(config["url"], str):
            raise ValueError(f"Invalid URL format for model: {service}")

        return OpenAI(base_url=config["url"], api_key=config["api_key"])

def ask_model(messages, service="openai", stream=False):
    """
    Function to interact with a specified AI model and handle the response.
    This function sends a list of messages to a selected AI service and model
    to get a chat completion response. Responses can either be streamed or
    returned as a single output.

    Parameters:
        messages (list): A list of message objects to be sent to the model.
        service (str): The name of the AI service to use. Default is "openai".
        stream (bool): Whether to stream the AI model's response in chunks.

    Returns:
        None
    """
    client = get_client_for_model(service)
    model = MODEL_CONFIGS[service]["model"]
    print(f"Using service: {service} and model: {model}, please wait...")
    resp = client.chat.completions.create(model=model, messages=messages, stream=stream)
    if stream:
        try:
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="")
        except Exception as e:
            print(f"\nError during streaming: {e}")
    else:
        print(resp.choices[0].message.content)

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

prompts = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt + question}
]

ask_model(prompts, service="openai", stream=False)
