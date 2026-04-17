from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from config import *
from contextlib import asynccontextmanager
from langchain_query import load_vectorstore, load_llm, build_rag_chain, define_documents


class Query(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = load_vectorstore()
    app.state.llm = load_llm()
    app.state.documents = define_documents(app.state.db)

    app.state.chain = build_rag_chain(app.state.db, app.state.llm, app.state.documents)

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
def ask_question(request: Query):
    result = app.state.chain.invoke({"input": request.query})
    return result