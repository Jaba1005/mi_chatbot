import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Cargar tu clave de API
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Inicializar el LLM de Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=GROQ_API_KEY
)

# 3. Cargar el modelo de embeddings local
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Conectar a la base de datos ChromaDB EXISTENTE
vector_store = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings,
    collection_name="mis_programas" # ¡Crucial para que encuentre tus PDFs!
)

# 5. Configurar la plantilla
PROMPT_TEMPLATE = '''Eres un asistente legal experto en reglamentos y normas de la Fundación Universitaria Konrad Lorenz.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información.
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# 6. Función principal
def rag_pipeline(pregunta, k=5):
    # Buscar en ChromaDB
    retriever_k = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever_k.invoke(pregunta)
    
    # --- DIAGNÓSTICO EN TERMINAL ---
    print(f"\n🔍 [DEBUG] Buscando: '{pregunta}'")
    print(f"📄 Fragmentos encontrados: {len(docs)}")
    if len(docs) > 0:
        print(f"💡 Primer fragmento: {docs[0].page_content[:100]}...")
    else:
        print("⚠️ ALERTA: ChromaDB devolvió 0 fragmentos. Revisa la ruta o el collection_name.")
    print("-" * 40)
    
    # Ensamblar el contexto
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    
    # Generar la respuesta
    prompt_con_rag = prompt_template.invoke({"context": contexto, "question": pregunta})
    texto_con_rag = llm.invoke(prompt_con_rag).content
    
    return texto_con_rag