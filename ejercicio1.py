import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Preparar las credenciales
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


def resumidor_pirata():
    # 2. Inicializar el LLM (Usamos Llama 3 por su velocidad)
    llm = ChatGroq(
        temperature=0.7,  # Un poco de creatividad para el estilo pirata
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    # 3. Definir la "Personalidad" y la "Tarea" (Prompt Engineering)
    # El System Message define el comportamiento; el Human Message es el dato.
    system_msg = "Eres un pirata de los siete mares experto en leyes educativas. Tu misión es resumir textos aburridos en 3 puntos clave usando jerga pirata."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Resume este texto en 3 puntos clave: {texto_a_resumir}")
    ])

    # 4. Crear la cadena simple (Chain)
    chain = prompt | llm

    # 5. El texto "aburrido" de prueba
    texto_ejemplo = """
    El módulo de Proyecto de desarrollo de aplicaciones Web tiene una duración total 
    de 40 horas y se imparte en el segundo curso del ciclo. Su evaluación se 
    realizará una vez cursados el resto de módulos profesionales y requiere la 
    elaboración de un producto final que integre las competencias adquiridas.
    """

    # 6. Ejecutar
    print("--- 🏴‍☠️ Procesando botín de información... ---\n")
    respuesta = chain.invoke({"texto_a_resumir": texto_ejemplo})

    print(respuesta.content)


if __name__ == "__main__":
    resumidor_pirata()