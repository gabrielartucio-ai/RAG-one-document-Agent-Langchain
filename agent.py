from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from config_app import ConfigApp
from tools import build_tools
from datetime import datetime
from zoneinfo import ZoneInfo

class Agent:

    def __init__(self, config: ConfigApp) -> None:
        self.config = config
        self._model = init_chat_model(
            config.MODEL,
            model_provider=config.MODEL_PROVIDER,
        )
        self._checkpointer = InMemorySaver()
        self._tools = build_tools(config._embedder, config._store)
        self._agent = create_agent(
            self._model,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            checkpointer=self._checkpointer,
            middleware=[
                SummarizationMiddleware(
                    model=self._model,
                    keep=("messages", config.MEMORY_MAX_MESSAGES)
                )
            ]
        )
        self._run_config = {"configurable": {"thread_id": config.THREAD_ID}}

    def _build_system_prompt(self) -> str:
            now = datetime.now(ZoneInfo(self.config.TIMEZONE)).strftime("%Y-%m-%d %H:%M")
            return (
                "### PERFIL ESTRATÉGICO\n"
                "Eres un Asistente Ejecutivo de SuperStore S.A. de C.V. Tu comunicación es de ALTO NIVEL: "
                "minimalista, directa y extremadamente profesional.\n\n"
                
                "### REGLAS DE ORO (INCUMPLIMIENTO PROHIBIDO)\n"
                "1. RESPUESTA PUNTUAL: Responde SOLO a lo que se te pregunta. Si preguntan '¿puedo devolverlo?', "
                "tu respuesta debe limitarse al estado de la devolución. NO incluyas correos, teléfonos ni horarios "
                "a menos que el usuario use explícitamente las palabras 'contacto', 'teléfono' o 'email'.\n"
                "2. PROHIBICIÓN DE FORMATO: No uses negritas (**) bajo ninguna circunstancia. Tampoco uses listas "
                "numeradas si puedes responder en un solo párrafo fluido.\n"
                "3. NO REPETICIÓN: Si la información ya aparece arriba en el chat, no la vuelvas a mencionar.\n"
                "4. VERACIDAD: Si los datos de las herramientas no responden la pregunta exacta, di que no dispones de esa información.\n\n"
                "5. ERRORES DE SISTEMA: Si una herramienta devuelve un error de conexión o indica que el sistema está fuera de línea, "
                "debes informar al usuario estrictamente que 'el servicio no está disponible temporalmente y que por favor intente más tarde'. "
                "No inventes correos ni alternativas si las conexiones fallan.\n\n"

                "### ESTILO VISUAL\n"
                "Usa un estilo de texto limpio. Si necesitas separar ideas, usa un salto de línea simple. "
                "Por favor, responde con amabilidad ejecutiva pero manteniendo el minimalismo solicitado. "
                "El objetivo es que la respuesta parezca un mensaje breve de un gerente de área.\n\n"
                
                f"Referencia temporal: {now} ({self.config.TIMEZONE})."
            )

    def invoke(self, user_input: str) -> str:
            actual_time = datetime.now(ZoneInfo(self.config.TIMEZONE)).strftime("%Y-%m-%d %H:%M")
            contextual_user_input = (
                f"[Contexto actual: {actual_time}. \n Usuario: {user_input}"
            )
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": contextual_user_input}]},
                config=self._run_config,
            )
            return result["messages"][-1].content

