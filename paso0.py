from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modelo de embeddings local — corre en CPU, se descarga ~120 MB la primera vez
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"[OK] Embeddings locales cargados: {EMBEDDING_MODEL}")

# Corpus de ejemplo
reglamento = [
    "Los alumnos con promedio superior a 4.5 reciben un incentivo económico.",
    "El abandono de los estudios sin previo aviso causa sanción administrativa.",
    "Se pueden pedir exámenes extraordinarios si hay una causa médica comprobada.",
    "La universidad ofrece apoyo financiero para proyectos de investigación.",
]

corpus_embeddings = embeddings.embed_documents(reglamento)
query = "ayuda de dinero por notas excelentes"
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], corpus_embeddings)
indices_ordenados = np.argsort(scores[0])[::-1]

print(f"\nConsulta: \"{query}\"\n")
for idx in indices_ordenados:
    barra = "█" * int(scores[0][idx] * 20)
    print(f"  {scores[0][idx]:.4f} {barra}")
    print(f"  {reglamento[idx]}\n")