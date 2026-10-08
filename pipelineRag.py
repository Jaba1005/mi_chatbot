import os
import time
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from embeddings_onnx import MultilingualOnnxEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-120b"

def log(mensaje):
    # flush=True para que aparezca en los logs de Render al instante
    print(f"[rag {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


if not GROQ_API_KEY:
    log("ADVERTENCIA: falta la variable de entorno GROQ_API_KEY")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    api_key=GROQ_API_KEY,
    timeout=60,
    max_retries=1
)

# Debe ser el mismo modelo con el que se indexó la base (ver paso4.py)
log("Cargando modelo de embeddings...")
embeddings_model = MultilingualOnnxEmbeddings()
# Primer embedding al arrancar, así la primera pregunta no espera la carga
embeddings_model.embed_query("calentamiento")
log("Modelo de embeddings listo")

vector_store = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings_model,
    collection_name="mis_programas",
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
)
log(f"Chroma lista: {vector_store._collection.count()} fragmentos")

PROMPT_TEMPLATE = '''Eres un asistente legal experto en reglamentos y normas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def _fuente(d):
    # La base se indexó en Windows: normalizamos "\" para que basename funcione en Linux
    fuente = os.path.basename(d.metadata.get("source", "?").replace("\\", "/"))
    pagina = d.metadata.get("page_label", d.metadata.get("page", "?"))
    return f"[Fuente: {fuente} — Pág. {pagina}]"


def rag_pipeline(pregunta, k=10):
    log(f"Pregunta recibida: {pregunta[:80]!r}")
    vector = embeddings_model.embed_query(pregunta)
    log("1/3 embedding de la pregunta listo")

    docs = vector_store.similarity_search_by_vector(vector, k=k)
    log(f"2/3 Chroma devolvió {len(docs)} fragmentos")

    contexto = "\n\n---\n\n".join(f"{_fuente(d)}\n{d.page_content}" for d in docs)

    prompt_con_rag = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta = llm.invoke(prompt_con_rag).content
    log("3/3 Groq respondió")
    return respuesta
