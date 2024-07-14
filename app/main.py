from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.langchain_config import configure_langchain, create_conversational_chain
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI()

# Definindo os domínios permitidos
origins = [
    "https://matheusgmaia33--hack-five.deco.site",
    "https://outrodominio.deco.site",
    "https://terceirodominio.deco.site",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar LangChain
vector_store = configure_langchain()
conversational_chain = create_conversational_chain(vector_store)

class ChatRequest(BaseModel):
    message: str
    history: list

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    user_history = request.history
    response = conversational_chain.run(question=user_message, chat_history=user_history)
    return ChatResponse(response=response)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Pet Adoption Bot API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
