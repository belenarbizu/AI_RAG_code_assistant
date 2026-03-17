from fastapi import FastAPI
from pydantic import BaseModel
from src.query import rag_pipeline

app = FastAPI()


class Query(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: Query):
    answer = rag_pipeline(request.query)
    return {"answer": answer}