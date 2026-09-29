import os
import shutil
from langchain_chroma import Chroma
from openai import embeddings

# 1. Importamos las variables de tu archivo anterior
from paso3 import chunks, embeddings_model

# Definimos el nombre del modelo solo para usarlo en el print de abajo
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PERSIST_DIR = "./chroma"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"Base de datos anterior eliminada: {PERSIST_DIR}")

print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
print("(Embeddings generados localmente con Sentence Transformers)")

# 4. Conectar a la base de datos ChromaDB EXISTENTE
vector_store = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings,
    collection_name="mis_programas" # Emoĩ ko línea
)

total = vector_store._collection.count()
print()
print("[OK] Base vectorial creada exitosamente!")
print(f"  Ubicación:             {PERSIST_DIR}/")
print(f"  Fragmentos indexados:  {total}")
print(f"  Modelo embeddings:     {EMBEDDING_MODEL} (local, CPU)")
print(f"  Dimensión de vectores: 384")

# Cargar base vectorial existente sin re-procesar (útil para reanudar)
# vector_store = Chroma(
#     persist_directory=PERSIST_DIR,
#     embedding_function=embeddings_model
# )
# print(f"Base vectorial cargada: {vector_store._collection.count()} fragmentos")

coleccion = vector_store._collection
print("Información de la colección ChromaDB:")
print(f"  Nombre:              {coleccion.name}")
print(f"  Total documentos:    {coleccion.count()}")
print(f"  Metadata:            {coleccion.metadata}")