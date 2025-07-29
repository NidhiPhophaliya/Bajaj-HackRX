from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class JustificationItem(BaseModel):
    clause: str
    reason: str

class QueryResponse(BaseModel):
    decision: str
    amount: str
    justification: List[JustificationItem]
