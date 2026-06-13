# ── Clients ──────────────────────────────────
from langchain_openai import AzureOpenAIEmbeddings
import logging

import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def embedding_client():
    try:
        return AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    max_retries=3)

    except Exception as e:
        logger.error(f"Failed to build Azure OpenAI clients: {e}")
        raise
    