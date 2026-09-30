import os
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from embeddings_onnx import MultilingualOnnxEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-120b"

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    api_key=GROQ_API_KEY
)

# Debe ser el mismo modelo con el que se indexó la base (ver paso4.py)
embeddings_model = MultilingualOnnxEmbeddings()
# Primer embedding al arrancar, así la primera pregunta no espera la carga
embeddings_model.embed_query("calentamiento")

vector_store = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings_model,
    collection_name="mis_programas",
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
)

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
    retriever_k = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever_k.invoke(pregunta)

    contexto = "\n\n---\n\n".join(f"{_fuente(d)}\n{d.page_content}" for d in docs)

    prompt_con_rag = prompt_template.invoke({"context": contexto, "question": pregunta})
    return llm.invoke(prompt_con_rag).content
