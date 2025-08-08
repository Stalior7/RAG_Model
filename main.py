from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn

# ✅ Import functions from handler
from model.handler import process_document_if_new, query_document_by_hash

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Pydantic model for request
class QueryRequest(BaseModel):
    documents: Optional[str] = None  # Optional file URL
    questions: List[str]

# ✅ Default document URL (fallback if no file is given)
DEFAULT_DOC_URL = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"  # 🔁 Replace with real link

@app.post("/hackrx/run")
async def run_query(request: QueryRequest):
    try:
        # ✅ Use given doc or fallback to default
        doc_url = request.documents if request.documents else DEFAULT_DOC_URL

        # ✅ Get or embed doc (returns hash)
        doc_hash = process_document_if_new(doc_url)

        # ✅ Process all questions in parallel
        answers = await query_document_by_hash(request.questions, doc_hash)

        # ✅ Format output
        results = {q: a for q, a in zip(request.questions, answers)}
        return {"status": "success", "answers": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
