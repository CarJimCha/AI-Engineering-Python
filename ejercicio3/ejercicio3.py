import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")


def run_rag_debug():
    # 1. Configuración
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
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

    # 3. Diseño del Prompt
    prompt = ChatPromptTemplate.from_template("""
    Responde basándote solo en el contexto proporcionado.
    Si no encuentras la respuesta en el contexto, di que no lo sabes.

    Contexto: {context}
    Pregunta: {input}
    """)

    # 4. Creación de la Cadena
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # 5. Ejecución con Auditoría
    #pregunta = "¿Cual es la duración del módulo Proyecto de desarrollo de aplicaciones Web?"
    pregunta = "¿Un sistema de recomendación de películas como el de Netflix analiza las películas que has visto en el pasado para sugerirte nuevas. ¿Qué concepto describe mejor cómo \"aprende\" el sistema?"

    print(f"\n--- Preguntando: {pregunta} ---")

    response = rag_chain.invoke({"input": pregunta})

    # --- PARTE CLAVE PARA LA CLASE: IMPRIMIR EL CONTEXTO ---
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