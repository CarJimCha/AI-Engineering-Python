from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def biblioteca_inteligente():
    # 1. Cargamos el modelo de embeddings (gratis/local)
    # Este modelo convierte las frases en vectores de 384 dimensiones
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Nuestra "Base de Datos" de la biblioteca
    libros = [
        "Harry Potter: Un niño huérfano descubre que es mago y asiste a una escuela de hechicería.",
        "El Señor de los Anillos: Un hobbit debe destruir un anillo de poder en un mundo de fantasía épica.",
        "Crónica de una muerte anunciada: El relato de un asesinato planeado que todo el pueblo conocía.",
        "Cien años de soledad: La historia de varias generaciones de la familia Buendía en el pueblo de Macondo.",
        "Steve Jobs: Biografía del fundador de Apple y su revolución en la tecnología personal."
    ]

    # 3. Indexación: Creamos el índice vectorial
    print("--- 🧙 Generando mapa de conocimientos de la biblioteca... ---")
    vector_db = FAISS.from_texts(libros, embeddings)

    # 4. Prueba de búsqueda semántica
    # NOTA: No usamos las palabras "Harry", "Potter", ni "Hechicería".
    query = "un chico sin padres que hace trucos de magia"

    print(f"\n🔍 Buscando por concepto: '{query}'")

    # Recuperamos el resultado más cercano (k=1)
    resultado = vector_db.similarity_search(query, k=1)

    print("\n📖 Recomendación del Bibliotecario:")
    print(f"-> {resultado[0].page_content}")


if __name__ == "__main__":
    biblioteca_inteligente()