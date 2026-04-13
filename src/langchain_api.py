from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.config import *
from contextlib import asynccontextmanager
from src.langchain_query import load_vectorstore, load_llm, build_rag_chain


class Query(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = load_vectorstore()
    app.state.llm = load_llm()
    app.state.chain = build_rag_chain(app.state.db, app.state.llm)

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
def ask_question(request: Query):
    result = build_rag_chain(app.state.db, app.state.llm).invoke({"input": request.query})
    return result