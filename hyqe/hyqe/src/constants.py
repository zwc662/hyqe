"""Set of constants."""

DEFAULT_TEMPERATURE = 0.1
DEFAULT_HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ENDPOINT_MODEL = "gpt-3.5-turbo"
DEFAULT_ENDPOINT_ENCODER = 'text-embedding-3-large'
DEFAULT_ENDPOINT_URL = "https://api.openai.com/v1"

DEFAULT_CONTEXT_WINDOW = 3900  # tokens
DEFAULT_NUM_OUTPUTS = 1024 #512  # tokens
DEFAULT_MAX_NEW_TOKEN = 1024 #512  # tokens
DEFAULT_NUM_INPUT_FILES = 10  # files

DEFAULT_EMBED_BATCH_SIZE = 10

DEFAULT_CHUNK_SIZE = 1024  # tokens
DEFAULT_CHUNK_OVERLAP = 20  # tokens
DEFAULT_SIMILARITY_TOP_K = 2
DEFAULT_IMAGE_SIMILARITY_TOP_K = 2

# NOTE: for text-embedding-ada-002
DEFAULT_EMBEDDING_DIM = 1536

# context window size for llm predictor
COHERE_CONTEXT_WINDOW = 2048
AI21_J2_CONTEXT_WINDOW = 8192


TYPE_KEY = "__type__"
DATA_KEY = "__data__"
VECTOR_STORE_KEY = "vector_store"
IMAGE_STORE_KEY = "image_store"
GRAPH_STORE_KEY = "graph_store"
INDEX_STORE_KEY = "index_store"
DOC_STORE_KEY = "doc_store"


DEFAULT_ENDPOINT_SYSTEM_PROMPT = """
You are an AI assistant. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Don't plainly replicate the given instruction.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language
"""



DEFAULT_SYSTEM_PROMPT = """<s>[INST]\nYou are an AI assistant. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Don't plainly replicate the given instruction.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.

The user prompt is as follows:\n\n{}[/INST]</s>
"""
