from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PersonalChatBot import PersonalChatBot


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = PersonalChatBot()
    yield


app = FastAPI(title="Anwaar's Personal Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Anwaar's Personal Assistant"}


@app.get("/chat", response_model=ChatResponse)
def chat(query: str = Query(..., description="User's message")):
    bot: PersonalChatBot = app.state.bot

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = bot.invoke_chain(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bot error: {str(e)}")

    return ChatResponse(response=result)


# Portfolio seedha root pe serve hoga
@app.get("/")
def serve_portfolio():
    return FileResponse("./static/portfolio.html")


# Static files (CSS, JS, images agar future me add karo)
app.mount("/static", StaticFiles(directory="./static"), name="static")