import os

class ConfigApp:
    TIMEZONE: str = "America/Montevideo"
    MODEL: str = "qwen/qwen3-32b"
    MODEL_PROVIDER: str = "groq"
    MEMORY_MAX_MESSAGES: int = 20
    THREAD_ID: str = "1"
    EXIT_COMMANDS: set[str] = {"salir", "exit", "quit"}
    AGENT_DESCRIPTION: str = "RAG Assistant"

    def __init__(self) -> None:
        self.GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
        self.SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
        self._validate_config()

    def _validate_config(self) -> None:
        """Verifica que las variables críticas existan al arrancar."""
        missing = [k for k, v in self.__dict__.items() if v is None]
        if missing:
            raise ValueError(f"Faltan variables de entorno críticas: {', '.join(missing)}")      
        

