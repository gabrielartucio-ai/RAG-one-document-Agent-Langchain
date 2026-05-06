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
            "Eres un agente de servicio al cliente del área de devoluciones. "
            "Los usuarios te contactarán con dudas relacionadas a las devoluciones. "
            "Sé conciso y breve. "
            "Los resultados de las tools son datos internos: nunca los traduzcas "
            "ni comentes sus palabras literales, solo usa su contenido para "
            "elaborar tu respuesta. "
            f"Fecha y hora de referencia: {now} ({self.config.TIMEZONE})."
        )

    def invoke(self, user_input: str) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=self._run_config,
        )
        return result["messages"][-1].content

