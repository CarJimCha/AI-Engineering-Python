import os
from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# --- IMPORTACIONES ESTÁNDAR (Verificadas para 2026) ---
# --- 1. AGENTES Y HERRAMIENTAS (Rutas Modernas) ---
from langchain_core.tools import create_retriever_tool

# --- 2. MOTOR LLM (GROQ) ---
from langchain_groq import ChatGroq

# --- 3. PROMPTS Y MEMORIA (CORE) ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 4. CARGA DE DATOS Y VECTORES (COMMUNITY) ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 5. MODELOS DE TEXTO (HUGGINGFACE) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def configurar_asistente():
    # --- 1. INGESTA DE DATOS (Vital para el retriever) ---
    if not os.path.exists("normativa"):
        os.makedirs("normativa")
        print("⚠️ Crea la carpeta 'normativa' y pon tus PDFs dentro.")
        return None

    print("--- 📂 Procesando documentos... ---")
    loader = PyPDFDirectoryLoader("normativa/")
    docs = loader.load()

    # En tu función de ingesta:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Aquí creamos la variable vector_db LOCALMENTE
    vector_db = FAISS.from_documents(chunks, embeddings)

    # --- 2. CONFIGURACIÓN DEL RETRIEVER ---
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})

    # --- 3. CONFIGURACIÓN DE LA TOOL ---
    tool = create_retriever_tool(
        retriever=retriever,
        name="buscador",
        description="Consulta este buscador para cualquier duda sobre módulos, horas y normativa del ciclo."
    )
    tools = [tool]

    # --- 4. CEREBRO Y PROMPT ---
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    system_msg = (
        "Eres el Asistente Oficial del Ciclo Formativo. "
        "REGLAS CRÍTICAS: "
        "1. Si el usuario te da las gracias, te dice 'Perfecto' o se despide, "
        "   NO uses ninguna herramienta. Responde amablemente y termina. "
        "2. Si la pregunta es sobre horas o módulos, usa 'buscador' UNA SOLA VEZ. "
        "3. Lee TODO el contexto que te dé el buscador antes de responder. "
        "Si ves varios números (ej. 128 y 64), explica a qué corresponde cada uno."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- 5. AGENTE Y MEMORIA ---
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=False,
        max_iterations=3,  # <--- Fuerza a que se detenga tras 5 intentos
        early_stopping_method="force"

    )
    demo_history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: demo_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# 4. INTERFAZ (Tu bucle de chat está perfecto)
def chat_asistente():
    asistente = configurar_asistente()
    if not asistente: return

    print("\n🎓 BIENVENIDO AL ASISTENTE DEL CICLO")
    config = {"configurable": {"session_id": "alumno_1"}}

    while True:
        usuario = input("👤 Tú: ")
        if usuario.lower() in ["salir", "exit"]: break

        response = asistente.invoke({"input": usuario}, config=config)
        print(f"🤖 Asistente: {response['output']}\n")


if __name__ == "__main__":
    chat_asistente()
