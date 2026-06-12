import os
from langchain_qdrant import QdrantVectorStore
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utility.retry_calling import with_retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_vectorDatabase(embedding):
    try:
        return QdrantVectorStore.from_existing_collection(
            embedding=embedding,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION"),
        )
    except Exception as e:
        logger.error(f"Failde to load vector databse :{e}")

def query(question: str,embedding,llm):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        vectorstore = load_vectorDatabase(embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        logger.error(f"Retriever setup failed: {e}")
        raise

    prompt = ChatPromptTemplate.from_template("""
Answer using only the context below.
If you don't know, say "I don't know."

Context: {context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        answer = with_retry(lambda: chain.invoke(question))
        logger.info(f"Query answered successfully.")
        return answer
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}")
        raise
