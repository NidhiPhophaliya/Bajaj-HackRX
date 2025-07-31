from fastapi import FastAPI, HTTPException, Request, Header
from app.models.schema import QueryRequest, QueryResponse, JustificationItem
from app.utils.search import SemanticSearch
from app.utils.llm_decider import generate_decision
import json, os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
app = FastAPI()

@app.get("/")
def health():
    return {"status": "HackRx API running 🚀"}


@app.post("/hackrx/run", response_model=QueryResponse)
def run_handler(request: Request, payload: QueryRequest, authorization: str = Header(None)):
    search_engine = SemanticSearch("app/data/chunks.csv")
    if not authorization or not authorization.startswith("Bearer ") or authorization.split()[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        results_df = search_engine.search(payload.query)
        top_chunks = results_df['text'].tolist()
        raw_output = generate_decision(payload.query, top_chunks)
        parsed = json.loads(raw_output)

        justification_items = [JustificationItem(**j) for j in parsed['justification']]
        return QueryResponse(
            decision=parsed['decision'],
            amount=parsed['amount'],
            justification=justification_items
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
