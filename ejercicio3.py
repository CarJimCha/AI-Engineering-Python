import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq  # <-- El nuevo motor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# 1. CARGAR CONFIGURACIÓN
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")


def run_rag_groq():
    print("--- Configurando Motor Groq y Embeddings locales ---")

    # El Cerebro: Groq (Usamos Llama 3.3 de 70B, que es potentísimo y gratis)
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_key
    )

    # El Traductor: HuggingFace (Gratis en tu PC)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. CARGAR PDF
    archivo = "politicas_empresa.pdf"  # O tu normativa de ciclos
    if not os.path.exists(archivo):
        print(f"Error: No falta el archivo {archivo}")
        return

    print(f"--- Leyendo {archivo} ---")
    loader = PyPDFLoader(archivo)
    docs = loader.load()

    # Trocear el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 4. BASE DE DATOS VECTORIAL
    print("--- Indexando en FAISS ---")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. PROMPT
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente experto. Responde basándote solo en el contexto:
    {context}

    Pregunta: {input}
    """)

    # 6. CADENA RAG
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # 7. RESULTADO
    pregunta = "¿Cuántos días de vacaciones tengo y qué es el Código Nebulosa?"
    print(f"--- Consultando a Groq Cloud (Velocidad LPU)... ---")

    response = rag_chain.invoke({"input": pregunta})

    print("\n--- RESPUESTA FINAL ---")
    print(response["answer"])


if __name__ == "__main__":
    run_rag_groq()