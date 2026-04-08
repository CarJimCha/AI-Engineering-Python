import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# Asegúrate de que en tu .env la variable se llame GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

def run_rag_debug():
    # 1. Configuración

    # 2. Inicializar el LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # O "gemini-2.5-pro" si quieres más potencia
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Carga
    archivo = "Rivas-Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf"
    if not os.path.exists(archivo):
        print(f"Error: No se encuentra {archivo}")
        return

    loader = PyPDFLoader(archivo)
    docs = loader.load()

    # ----- OPCIONES -----
    # Opción Original:
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Opción A
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

    # Opción B
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

    chunks = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 3. Diseño del Prompt (Versión para Gemini)
    prompt = ChatPromptTemplate.from_template("""
        Utiliza el siguiente contexto para responder a la pregunta del usuario. 
        Si la respuesta no está literal, intenta deducirla basándote en la información disponible.
        Si el contexto no tiene absolutamente nada que ver, indica qué temas se tratan en el texto.

        Contexto: {context}
        Pregunta: {input}
        """)

    # 4. Creación de la Cadena
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # El retriever actúa como el 'filtro' que decide qué 5 trozos (k=5 por defecto)
    # de los miles que hay en el PDF son los que Gemini debe leer.
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # 5. Ejecución con Auditoría
    # pregunta = " Un hospital implementa una IA para ayudar a los radiólogos. El KPI Reducción de la tasa de error diagnóstico en un 5%, es un ejemplo de:"
    pregunta = "¿Qué es un KPI y pon un ejemplo de la guía?"

    print(f"\n--- Preguntando: {pregunta} ---")

    response = rag_chain.invoke({"input": pregunta})

    # --- IMPRIMIR EL CONTEXTO ---
    print("\n" + "=" * 50)
    print("CONTEXTO RECUPERADO (Lo que el buscador encontró)")
    print("=" * 50)

    for i, doc in enumerate(response["context"]):
        page_number = doc.metadata.get('page', 'N/A')  # Obtener la página desde los metadatos
        print(f"\nFRAGMENTO {i + 1} (Página {page_number}):")
        print("-" * 30)
        print(doc.page_content)
        print("-" * 30)

    # Recorremos los documentos recuperados
    for i, doc in enumerate(response["context"]):
        print(f"\nFRAGMENTO {i + 1} (Página {doc.metadata.get('page', 'N/A')}):")
        print("-" * 30)
        print(doc.page_content)  # Aquí imprimimos el texto real del trozo
        print("-" * 30)

    print("\n" + "=" * 50)
    print("RESPUESTA FINAL DE LA IA")
    print("=" * 50)
    print(response["answer"])

    print("\nPágina/s usada/s en la respuesta:")
    for doc in response["context"]:
        page_number = doc.metadata.get('page', 'N/A')  # Obtener la página desde los metadatos
        print(f"- Información recuperada de la página: {page_number}")


    # Prueba A
    # ==================================================
    # RESPUESTA FINAL DE LA IA
    # ==================================================
    # Según el contexto proporcionado, un KPI (Key Performance Indicator o Indicador Clave de Desempeño) es una métrica que se utiliza para evaluar la eficiencia.
    #
    # Un ejemplo de KPI Operativo mencionado en la guía es la **reducción de la estancia**.
    #
    # Página/s usada/s en la respuesta:
    # - Información recuperada de la página: 108
    # - Información recuperada de la página: 113
    # - Información recuperada de la página: 44
    # - Información recuperada de la página: 44

    # La respuesta es, de hecho, coherente. Es cierto que los otros ejemplos no tenían tanto sentido, pero la respuesta final la veo razonable.


    # Prueba B
    # ==================================================
    # RESPUESTA FINAL DE LA IA
    # ==================================================
    # Un KPI (Key Performance Indicator o Indicador Clave de Rendimiento) es una métrica que se utiliza para medir la eficiencia de un sistema o proyecto.
    #
    # Un ejemplo de KPI operativo mencionado en la guía es:
    #
    # *   **La reducción del tiempo de inactividad de los quirófanos entre cirugías.**
    #
    # Página/s usada/s en la respuesta:
    # - Información recuperada de la página: 82
    # - Información recuperada de la página: 19
    # - Información recuperada de la página: 113
    # - Información recuperada de la página: 15

    # Tarda algo más y el contexto era demasiado largo, pero la respuesta es válida.

if __name__ == "__main__":
    run_rag_debug()