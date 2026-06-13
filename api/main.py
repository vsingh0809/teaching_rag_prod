import logging
import sys
import os
from dotenv import load_dotenv
from ingestion.ingest import ingest
from retrieval.retriever import query
from clients.embeddings import embedding_client
from clients.llm import llm_client
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException, UploadFile, File
import shutil
import tempfile
from fastapi.responses import JSONResponse
from models.QueryResponse import QueryResponse
from models.QueryRequest import QueryRequest

# Setup basic logging for production monitoring
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app_state={}

@asynccontextmanager
async def lifespan(app:FastAPI):
    log.info("Starting up - initializing clients...")
    try:
        app_state["embeddings"]=embedding_client()
        app_state["llm"]=llm_client()
        log.info("Clients ready")
    except Exception as e:
        log.critical(f"startup failed:{e}")
        raise    
    yield
    #Shutdown
    app_state.clear()
    log.info("Shutdown complete")

app= FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan
)    

@app.get("/health")
async def health():
    return{"status":"healthy","clients_ready":bool(app_state)}

#-------INGEST----------------------------------
@app.post("/ingest")
async def ingest_file(file:UploadFile=File(...)):
    #validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="only pdf files accepted")
    #Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file,tmp)
            tmp_path=tmp.name
    except Exception as e:
        log.error(f"failed to save upload:{e}")
        raise HTTPException(status_code=500,detail="Failed to save uploaded file.")

    try:
        ingest(tmp_path,app_state["embeddings"])
        return JSONResponse(content={
            "status":"success",
            "message":f"{file.filename} ingested successfully."
        })  
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed.")
    finally:
        # Always clean temp file — even if ingestion crashes
        os.unlink(tmp_path)  

@app.post("/query",response_model=QueryResponse)
async def post_query(request:QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="questions cant be empty")
    try:
        answer=query(request.question,app_state["embeddings"],app_state["llm"])
        return QueryResponse(answer=answer)
    except Exception as e:
        log.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Query failed.")
        

def validate_env():
    
    required = [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION", "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_CHAT_DEPLOYMENT", "QDRANT_URL",
        "QDRANT_API_KEY", "QDRANT_COLLECTION",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}")


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage:\n  python main.py ingest <pdf>\n  python main.py query")
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            log.error("Provide PDF: python main.py ingest file.pdf")
            sys.exit(1)
        try:
            ingest(sys.argv[2],embedding_client())
        except Exception as e:
            log.error(f"Ingestion failed: {e}")

    elif command == "query":
        log.info("Query mode — type 'exit' to quit")
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q.lower() == "exit":
                    break
                if not q:
                    continue
                answer = query(q,embedding_client(),llm_client())
                print(f"\n💬 {answer}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                log.error(f"Query failed: {e}")
    else:
        log.error(f"Unknown command: {command}")