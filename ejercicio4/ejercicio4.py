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

load_dotenv()

def configurar_asistente():
    if not os.path.exists("normativa"):
        os.makedirs("normativa")
        return None

    # Mensaje discreto de carga
    sys.stdout.write("--- 📂 Indexando normativa, espera un momento... ")
    sys.stdout.flush()

    loader = PyPDFDirectoryLoader("normativa/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    sys.stdout.write("✅ ¡Listo!\n")

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    tool = create_retriever_tool(
        retriever=retriever,
        name="buscador_normativa",
        description="Consulta para buscar información sobre módulos, horas y duraciones."
    )
    tools = [tool]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
        timeout=None,
    )

    system_msg = (
        "Eres el Asistente Oficial del Ciclo Formativo. Responde de forma clara y amable. "
        "Si usas el buscador para dar un dato numérico (como horas), asegúrate de citar el módulo correctamente."
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
        verbose=False,  # <-- Oculta el razonamiento
        handle_parsing_errors=True,
        early_stopping_method="force"
    )

    demo_history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: demo_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def chat_asistente():
    asistente = configurar_asistente()
    if not asistente: return

    print("\n" + "=" * 40)
    print("🎓 SISTEMA DE CONSULTA EDUCATIVA v2.0")
    print("   Escribe 'salir' para finalizar")
    print("=" * 40 + "\n")

    config = {"configurable": {"session_id": "sesion_alumnos"}}

    while True:
        usuario = input("Tú: ")
        if usuario.lower() in ["salir", "exit"]: break

        response = asistente.invoke({"input": usuario}, config=config)

        # --- LÓGICA DE LIMPIEZA DE RESPUESTA (PARSING) ---
        raw_output = response.get('output', '')
        texto_limpio = ""

        # Si Gemini nos devuelve la respuesta en trozos (una lista)
        if isinstance(raw_output, list):
            for fragmento in raw_output:
                # Si el trozo es un diccionario (como el que tiene el signature)
                if isinstance(fragmento, dict) and 'text' in fragmento:
                    texto_limpio += fragmento['text']
                # Si el trozo es directamente el texto final
                elif isinstance(fragmento, str):
                    texto_limpio += fragmento
        else:
            # Si la respuesta ya es un texto normal y corriente
            texto_limpio = str(raw_output)

        # Imprimimos el resultado impecable
        print(f"Asistente: {texto_limpio}\n")


if __name__ == "__main__":
    chat_asistente()