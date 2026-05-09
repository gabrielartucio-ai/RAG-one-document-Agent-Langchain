import logging

from dotenv import load_dotenv
from config_app import ConfigApp
from agent import Agent
from gemini_embedder import GeminiEmbedder
from supabase_doc_store import SupabaseDocStore

from datetime import datetime
from zoneinfo import ZoneInfo

# --- Configuración de silencio de logs ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.basicConfig(level=logging.ERROR)
# ---------------------------------------------

class RagAgent:

    def __init__(self):
        load_dotenv()
        self.config = ConfigApp()
        embedder = GeminiEmbedder(gemini_api_key=self.config.GEMINI_API_KEY)
        store = SupabaseDocStore(database_url=self.config.SUPABASE_URL)
        self.agent = Agent(config=self.config, embedder=embedder, store=store)

    def _print_welcome(self) -> None:
        now = datetime.now(ZoneInfo(self.config.TIMEZONE)).strftime("%Y-%m-%d %H:%M")  # FIX: calculado localmente
        print("=" * 40)
        print(self.config.AGENT_DESCRIPTION)   
        print(f"Hoy es {now}")
        print("=" * 40)
        print("\nBienvenido. ¿En qué puedo ayudarte?")
        print(f"Escribe {' o '.join(self.config.EXIT_COMMANDS)} para terminar.\n")

    def run(self) -> None:
        self._print_welcome()
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in self.config.EXIT_COMMANDS:
                print("¡Hasta luego!")
                break
            response = self.agent.invoke(user_input)
            print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    RagAgent().run()

