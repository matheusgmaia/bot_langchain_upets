from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.langchain_config import configure_langchain, create_conversational_chain
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI()

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
    return {"messages": "Welcome to the Pet Adoption Bot API"}

if __name__ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)