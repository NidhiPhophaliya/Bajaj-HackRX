from fastapi import FastAPI, HTTPException, Request, Header
from app.models.schema import QueryRequest, QueryResponse, JustificationItem
from app.utils.search import SemanticSearch
from app.utils.llm_decider import generate_decision
from app.utils.embedder import Embedder
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
app = FastAPI()

# ✅ Load the embedder once from Drive
embedder = Embedder()
embedder.load_from_drive(
    index_url="https://drive.google.com/uc?id=1GOSzA4PiEsDZupMEeNsuIEhKpbRMWgxl",
    metadata_url="https://drive.google.com/uc?id=1MPkhB5L0TkXNivb1SjRlhYejDpP9Mp6v"
)

# ✅ Semantic Search wrapper
search_engine = SemanticSearch(embedder)

@app.post("/debug/test")
def debug(payload: dict):
    return {"echo": payload}

@app.get("/")
def health():
    return {"status": "HackRx API running 🚀"}

@app.head("/")
def health_head():
    return

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    body = await request.body()
    print("📥 RAW Body:", body.decode("utf-8"))
    print("📥 Headers:", dict(request.headers))
    response = await call_next(request)
    return response

@app.post("/hackrx/run", response_model=QueryResponse)
def run_handler(request: Request, payload: QueryRequest, authorization: str = Header(None)):
    print("📩 Incoming query:", payload.query)

    if not authorization or not authorization.startswith("Bearer ") or authorization.split()[1] != API_KEY:
        print("❌ Unauthorized request")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        results_df = search_engine.search(payload.query)
        print(f"🔍 Found {len(results_df)} relevant chunks")

        if results_df.empty:
            raise HTTPException(status_code=404, detail="No relevant information found")

        top_chunks = results_df['text'].tolist()
        print("📄 Preview chunk:", top_chunks[0][:150])

        # ✅ Use LLM
        parsed = generate_decision(payload.query, top_chunks)

        justification_items = [JustificationItem(**j) for j in parsed.get('justification', [])]

        return QueryResponse(
            decision=parsed.get('decision', "No decision provided"),
            amount=parsed.get('amount', "N/A"),
            justification=justification_items
        )

    except Exception as e:
        print("🔥 Internal error:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
