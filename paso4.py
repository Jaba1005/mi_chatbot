import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Modelo multilingüe: entiende español (all-MiniLM-L6-v2 solo inglés)
# Debe ser EL MISMO que usa pipelineRag.py para consultar
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PDF_DIR = "pdfs"
PERSIST_DIR = "./chroma"
COLLECTION_NAME = "mis_programas"

# 1. Cargar los PDFs
documents = []
for pdf_file in sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")):
    pages = PyPDFLoader(os.path.join(PDF_DIR, pdf_file)).load()
    documents.extend(pages)
    print(f"  {pdf_file}: {len(pages)} páginas cargadas")

# 2. Partir en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
print(f"Fragmentos generados: {len(chunks)}")

# 3. Embeddings locales
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Borrar la base anterior y crear una nueva con los fragmentos
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"Base de datos anterior eliminada: {PERSIST_DIR}")

print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    collection_metadata={"hnsw:space": "cosine"}
)

coleccion = vector_store._collection
print()
print("[OK] Base vectorial creada exitosamente!")
print(f"  Ubicación:             {PERSIST_DIR}/")
print(f"  Colección:             {coleccion.name}")
print(f"  Fragmentos indexados:  {coleccion.count()}")
print(f"  Modelo embeddings:     {EMBEDDING_MODEL} (local, CPU)")
