from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import logging
import os
from .cleaning import clean_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest(pdf_path:str,embedding):
    if not os.path.exists(pdf_path):
        logger.error(f"file not found{pdf_path}")
        return

    try:
        logger.info(f"loading PDF{pdf_path}")
        loader=PyPDFLoader(pdf_path)
        docs=loader.load()
        print(f"Loaded {len(docs)} pages")

        docs=clean_documents(docs)

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=60
        )
        
        chunks=splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")



        logger.info("connecting to Qdrant client")
        vectorstore=QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION"),
        )
        logger.info(f"ingestion completed successfully with chunks{len(chunks)}")
        return vectorstore
    except Exception as e:
        logger.error(f"Ingestion pipeline failed:{e}")