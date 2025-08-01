from fastapi import FastAPI, HTTPException, Request, Header
from app.models.schema import QueryRequest, QueryResponse, JustificationItem
from app.utils.search import SemanticSearch
from app.utils.llm_decider import generate_decision
import json, os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
app = FastAPI()

@app.post("/debug/test")
def debug(payload: dict):
    return {"echo": payload}

@app.get("/")
def health():
    return {"status": "HackRx API running 🚀"}

# ✅ Add this to support HEAD / and avoid 405 errors
@app.head("/")
def health_head():
    return

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    body = await request.body()
    print("📥 RAW Incoming Body:", body.decode("utf-8"))
    print("📥 Headers:", dict(request.headers))
    response = await call_next(request)
    return response

@app.post("/hackrx/run", response_model=QueryResponse)
def run_handler(request: Request, payload: QueryRequest, authorization: str = Header(None)):
    print("📩 Incoming request payload:", payload.query)
    print("🔐 Authorization header:", authorization)
    print("🔑 Expected API_KEY from .env:", API_KEY)

    if not authorization or not authorization.startswith("Bearer ") or authorization.split()[1] != API_KEY:
        print("❌ Authorization failed")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        print("✅ Auth passed. Loading chunks...")
        search_engine = SemanticSearch("app/data/chunks.csv")

        results_df = search_engine.search(payload.query)
        print(f"🔎 Search results found: {len(results_df)} rows")

        if results_df.empty:
            print("⚠️ No chunks matched the query")
            raise HTTPException(status_code=404, detail="No relevant information found")

        top_chunks = results_df['text'].tolist()
        print("📄 Top chunk preview:", top_chunks[:1])

        raw_output = generate_decision(payload.query, top_chunks)
        print("🤖 LLM raw output:", raw_output)

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as je:
            print("❌ JSON parsing error:", je)
            raise HTTPException(status_code=500, detail="Invalid response from LLM")

        print("✅ Parsed JSON:", parsed)

        justification_items = [JustificationItem(**j) for j in parsed.get('justification', [])]
        print("🧾 Justification prepared")

        return QueryResponse(
            decision=parsed.get('decision', "No decision provided"),
            amount=parsed.get('amount', "N/A"),
            justification=justification_items
        )

    except Exception as e:
        print("🔥 Uncaught exception:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
