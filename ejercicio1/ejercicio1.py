import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Preparar las credenciales
load_dotenv()
# Asegúrate de que en tu .env la variable se llame GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

def traductor_tecnico():
    # 2. Inicializar el LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # 3. Definir la "Personalidad" y la "Tarea" (Prompt Engineering)
    system_msg = """Eres un traductor técnico que toma errores de código complejos y los explica de forma sencilla.
    Dependiendo del nivel del público, ajusta la complejidad de la explicación:
    - Si el público es un "niño de 5 años", usa analogías simples como juguetes o historias.
    - Si el público es un "CEO con prisa", sé breve y directo.
    - Si el público es un "estudiante de DAW", usa lenguaje técnico básico, pero comprensible.
    Tu tarea es explicar el error y dar una posible solución en una línea de código.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Explica el siguiente error técnico: {error_tecnico}. Explicalo de manera comprensible para un "
                  "{nivel_del_publico}. Propón una solución en una sola línea de código python.")
    ])

    # 4. Crear la cadena simple (Chain)
    chain = prompt | llm

    # 5. Valores de las variables
    error_tecnico = "NullPointerException"
    nivel_del_publico = "niño de 5 años"

    # 6. Ejecutar
    print("--- Procesando el error... ---\n")
    respuesta = chain.invoke({"error_tecnico": error_tecnico, "nivel_del_publico": nivel_del_publico})

    print("Explicación y solución propuesta:")
    print(respuesta.content)

if __name__ == "__main__":
    traductor_tecnico()