from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def filtro_candidatos():
    # Actualizo el modelo para que no sea tan literal y detecte al candidato trampa
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    perfiles_candidatos = [
        "Experto en Python y backend con 5 años de experiencia.",
        "Diseñador gráfico especializado en interfaces móviles.",
        "Desarrollador Full Stack con experiencia en aplicaciones web.",
        "Administrador de bases de datos con 3 años de experiencia en SQL y NoSQL.",
        "Camarero con experiencia sirviendo mesas y gestionando órdenes."  # Candidato Trampa
    ]
    vector_db = FAISS.from_texts(perfiles_candidatos, embeddings)

    print("--- ELECCIÓN DEL CANDIDATO IDEAL ---")
    query_vacante = "Buscamos un programador para crear servidores y lógica de negocio"

    # Buscamos los candidatos más similares a la vacante
    resultados = vector_db.similarity_search_with_score(query_vacante, k=5)

    print(f"Requisito: '{query_vacante}'\n")
    print(f"{'Candidato':<405} | {'Distancia (Menor es mejor)'}")
    print("-" * 45)

    for doc, score in resultados[:3]:
        # El 'score' es la distancia Euclidiana
        print(f"{doc.page_content:<40} | {score:.4f}")

    print("\n--- Candidato Trampa ---")
    doc_trampa, score_trampa = resultados[4]
    print(f"{doc_trampa.page_content:<40} | {score_trampa:.4f}")

if __name__ == "__main__":
    filtro_candidatos()