from dotenv import load_dotenv
from config_app import ConfigApp
from agent import Agent
from datetime import datetime
from zoneinfo import ZoneInfo

class RagAgent:

    def __init__(self):
        load_dotenv()
        self.config = ConfigApp()
        self.agent = Agent(config=self.config)

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
            print(f"Assistant: {response}\n")

if __name__ == "__main__":
    RagAgent().run()

