import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Preparar las credenciales
load_dotenv()
# Asegúrate de que en tu .env la variable se llame GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")

def resumidor_pirata():
    # 2. Inicializar el LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # 3. Definir la "Personalidad" y la "Tarea" (Prompt Engineering)
    system_msg = "Eres un pirata de los siete mares experto en leyes educativas. Tu misión es resumir textos aburridos en 3 puntos clave usando jerga pirata."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Resume este texto en 3 puntos clave: {texto_a_resumir}")
    ])

    # 4. Crear la cadena simple (Chain)
    chain = prompt | llm

    # 5. El texto de prueba
    texto_ejemplo = """
    El módulo de Proyecto de desarrollo de aplicaciones Web tiene una duración total 
    de 40 horas y se imparte en el segundo curso del ciclo. Su evaluación se 
    realizará una vez cursados el resto de módulos profesionales y requiere la 
    elaboración de un producto final que integre las competencias adquiridas.
    """

    # 6. Ejecutar
    print("--- Procesando botín de información con Gemini... ---\n")
    respuesta = chain.invoke({"texto_a_resumir": texto_ejemplo})

    print(respuesta.content)

if __name__ == "__main__":
    resumidor_pirata()