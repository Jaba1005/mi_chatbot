import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=GROQ_API_KEY
)

# Usamos la función nativa ligera de Chroma (cero consumo de RAM)
lightweight_ef = DefaultEmbeddingFunction()

vector_store = Chroma(
    persist_directory="./chroma",
    embedding_function=lightweight_ef,
    collection_name="mis_programas"
)

PROMPT_TEMPLATE = '''Eres un asistente académico experto en el reglamento institucional.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del estudiante: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def rag_pipeline(pregunta, k=5):
    retriever_k = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever_k.invoke(pregunta)
    
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    
    prompt_con_rag = prompt_template.invoke({"context": contexto, "question": pregunta})
    texto_con_rag = llm.invoke(prompt_con_rag).content
    
    return texto_con_rag