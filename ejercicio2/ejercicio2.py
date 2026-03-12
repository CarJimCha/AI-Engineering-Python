from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def biblioteca_inteligente_con_metricas():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Creamos una base de datos rápida con palabras para comparar
    # La diferencia entre "planeta" y "Planeta"
    # demuestra que el modelo es sensible a la capitalización,
    # aunque mantenga la cercanía semántica.
    palabras_test = ["gol", "planeta", "grada", "libro", "Planeta"]
    vector_db = FAISS.from_texts(palabras_test, embeddings)

    print("--- COMPARACIÓN DE DISTANCIAS SEMÁNTICAS ---")
    query_palabra = "futbol"

    # Buscamos las distancias de "futbol" contra nuestra base
    # k=4 para ver todas las palabras del test
    resultados = vector_db.similarity_search_with_score(query_palabra, k=5)

    print(f"Palabra de origen: '{query_palabra}'\n")
    print(f"{'Palabra destino':<15} | {'Distancia (Menor es mejor)':<25}")
    print("-" * 45)

    for doc, score in resultados:
        # El 'score' es la distancia Euclidiana
        print(f"{doc.page_content:<15} | {score:.4f}")

    print("\n" + "=" * 50)
    print("--- BUSCADOR DE LIBROS ---")
    libros = [
        "Harry Potter: Un niño huérfano descubre que es mago y asiste a una escuela de hechicería.",
        "El Señor de los Anillos: Un hobbit debe destruir un anillo de poder en un mundo de fantasía épica.",
        "Steve Jobs: Biografía del fundador de Apple y su revolución en la tecnología personal."
    ]
    vector_db_libros = FAISS.from_texts(libros, embeddings)

    query_libro = "un chico sin padres que hace trucos de magia"
    # Recuperamos el doc y su score
    doc_libro, score_libro = vector_db_libros.similarity_search_with_score(query_libro, k=1)[0]

    print(f"Buscando: '{query_libro}'")
    print(f"Recomendación: {doc_libro.page_content}")
    print(f"Confianza (Distancia): {score_libro:.4f}")

if __name__ == "__main__":
    biblioteca_inteligente_con_metricas()