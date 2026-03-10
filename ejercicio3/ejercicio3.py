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

    # 2. Carga (Cambiamos al archivo que tienes: fasciculo-2.pdf)
    archivo = "fasciculo-2.pdf"
    archivo = "Rivas-Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf"
    if not os.path.exists(archivo):
        print(f"Error: No se encuentra {archivo}")
        return

    loader = PyPDFLoader(archivo)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # 5. Ejecución con Auditoría
    #pregunta = "¿Cual es la duración del módulo Proyecto de desarrollo de aplicaciones Web?"
    pregunta = " Un hospital implementa una IA para ayudar a los radiólogos. El KPI Reducción de la tasa de error diagnóstico en un 5%, es un ejemplo de:"

    print(f"\n--- Preguntando: {pregunta} ---")

    response = rag_chain.invoke({"input": pregunta})

    # --- IMPRIMIR EL CONTEXTO ---
    print("\n" + "=" * 50)
    print("CONTEXTO RECUPERADO (Lo que el buscador encontró)")
    print("=" * 50)

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


if __name__ == "__main__":
    run_rag_debug()