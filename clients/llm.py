from langchain_openai import AzureChatOpenAI
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

def llm_client():
    try:
         return AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_retries=3
         )
    except Exception as e:
        logger.error(f"Failed to build Azure OpenAI clients: {e}")
        raise