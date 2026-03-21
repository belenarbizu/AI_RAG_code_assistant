from fastapi import FastAPI
from pydantic import BaseModel
from src.query import rag_pipeline, get_collection, load_llm, build_bm25_index
from sentence_transformers import SentenceTransformer
from src.config import *
from contextlib import asynccontextmanager
import bm25s


class Query(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = get_collection()
    data = app.state.collection.get(include=["documents"])
    app.state.documents = data["documents"]
    app.state.bm25 = build_bm25_index(app.state.documents)
    app.state.model = SentenceTransformer(EMBEDDING_MODEL)
    app.state.tokenizer, app.state.llm = load_llm()

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
def ask_question(request: Query):
    result = rag_pipeline(request.query, app.state.collection, app.state.bm25, app.state.documents, app.state.model, app.state.tokenizer, app.state.llm)
    return result