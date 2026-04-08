import os
import sys
from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# --- IMPORTACIONES ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

load_dotenv()

@tool
def consultar_calendario_examenes():
    """
    Consulta las fechas de los exámenes del ciclo formativo.
    Úsala cuando el usuario pregunte por fechas o exámenes.
    """
    return {
        "Proyecto Web": "15 de junio",
        "Recuperaciones": "20 de junio",
        "Fin de exámenes del tercer trimestre": "25 de mayo",
        "Comienzo de exámenes del tercer trimestre": "15 de mayo"
    }

def configurar_asistente():
    if not os.path.exists("normativa"):
        os.makedirs("normativa")
        print("Crea la carpeta 'normativa' y pon tus PDFs dentro.")
        return None

    sys.stdout.write("--- Indexando normativa... ")
    sys.stdout.flush()

    loader = PyPDFDirectoryLoader("normativa/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    sys.stdout.write("¡Listo! \n")

    retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    tool_normativa = create_retriever_tool(
        retriever=retriever,
        name="buscador_normativa",
        description="Consulta para buscar información oficial sobre el ciclo, módulos y horas."
    )
    tools = [tool_normativa, consultar_calendario_examenes]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_output_tokens=600,
        max_retries=2,
    )

    system_msg = (
        "Eres un asistente educativo especializado en el ciclo formativo. "
        "Tienes acceso a dos herramientas:\n"
        "- 'buscador_normativa': para consultar módulos, contenidos y normativa.\n"
        "- 'consultar_calendario_examenes': para consultar fechas de exámenes.\n\n"
        "Usa cada herramienta según corresponda. "
        "Si no necesitas herramientas, responde directamente de forma clara y amable."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

def limpiar_respuesta(salida_raw):
    """Extrae únicamente el texto de la respuesta de Gemini."""
    if isinstance(salida_raw, list):
        texto = ""
        for item in salida_raw:
            if isinstance(item, dict) and 'text' in item:
                texto += item['text']
            elif isinstance(item, str):
                texto += item
        return texto
    return str(salida_raw)

def chat_asistente():
    asistente = configurar_asistente()
    if not asistente: return

    print("\n" + "=" * 40)
    print("SISTEMA DE CONSULTA EDUCATIVA v3.0")
    print("   Escribe 'salir' para finalizar")
    print("=" * 40 + "\n")

    config = {"configurable": {"session_id": "sesion_docente"}}

    while True:
        usuario = input("Tú: ")
        if usuario.lower() in ["salir", "exit"]: break

        try:
            response = asistente.invoke({"input": usuario}, config=config)
            respuesta_final = limpiar_respuesta(response["output"])
            print(f"Asistente: {respuesta_final}\n")

        except Exception as e:
            print(f"Error en la comunicación: {e}")

if __name__ == "__main__":
    chat_asistente()