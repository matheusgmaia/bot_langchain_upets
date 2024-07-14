import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

PROMPT_BASE = """
Você é o Assistente UPets, um assistente virtual em um site de adoção de animais chamado uPets e deve formular uma única mensagem para ser apresentada ao Tutor e continuar a conversa de forma fuida.
Suas respostas devem ser diretas eficientes e sucintas.
Seu objetico é realizar o match ideal de Pet com base nas informações coletadas. 
Antes de sugerir um pet você deve ter as seguintes informações coletadas:
1. Gato ou cachorro?
2. Macho ou fêmea?
3. Pequeno ou grande porte?
4. Filhote ou aldulto?
5. Tem preferência por alguma cor?

(Após o Tutor ter decidido por um Pet, enfatize as políticas, peça um contato e informe que a equipe entrará em contato.)
(Evite perguntar por informações já coletadas)
(Você deve tratar o usuário como Tutor e sempre iniciar a resposta com "Assistente UPets:")
Segue o contexto. hitórico e última mensagem para consideração. 
"""

# Carregar variáveis de ambiente
load_dotenv()

# Carregar chave da API da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def configure_langchain():
    """Configura LangChain com os dados fornecidos."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    pets_documents = CSVLoader('./app/data/pets.csv').load()
    
    text_documents = TextLoader('./app/data/pets.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_documents_s = text_splitter.split_documents(text_documents)
    
    vector_store = Chroma.from_documents(pets_documents + text_documents_s, embeddings)
    
    return vector_store

def get_chat_model():
    """Retorna o modelo de linguagem configurado."""
    return OpenAI(model="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY)

def create_conversational_chain(vector_store):
    """Cria a cadeia de consulta conversacional."""
    chat_model = get_chat_model()

    custom_template = """
    
    Última mensagem do usuário: {question}
    
    Repita a última mensagem do usuário.
    """
    
    general_system_template = PROMPT_BASE + "\n----\n{context}\n----\n"

    general_user_template = "Histórico da Conversa: {chat_history} \n Última mensagem do Tutor: `{question}`"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    pt = PromptTemplate(input_variables=['chat_history', 'question'], template=custom_template) 

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        condense_question_prompt=pt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        get_chat_history=lambda h: h,
        verbose=True
    )

    return chain

# Configurar LangChain
vector_store = configure_langchain()
conversational_chain = create_conversational_chain(vector_store)
