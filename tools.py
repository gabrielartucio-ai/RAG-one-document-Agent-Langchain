from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def make_query_tool(embedder, store) -> list:

    def log_retry_attempt(retry_state):
            """Aviso que se imprime en consola antes de cada reintento."""
            tried = retry_state.attempt_number
            print(f"⚠️ [AVISO] Problema de conexión. Reintentando en breve... (Intento {tried})")

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=log_retry_attempt, 
        reraise=True 
    )
    def _safe_embed(query: str):
        """Genera el embedding con reintentos si falla la API de Gemini."""
        return embedder.embed_query(query)

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=log_retry_attempt, # <--- Esta es la clave
        reraise=True
    )
    def _safe_search(vector):
        """Realiza la búsqueda en Supabase con reintentos si falla la conexión."""
        return store.search(query_emb=vector)

    @tool(parse_docstring=True)
    def query_documents(query: str) -> dict:
        """Busca en la base de datos oficial sobre políticas, plazos y medios de contacto 
        (teléfono, email, horarios).

        Args:
            query: Términos de búsqueda en español.
        """

        try:
            query_vector = _safe_embed(query)
            hits = _safe_search(query_vector)
            return {
                "query": query,
                "matches": hits,
                "source": "PostgreSQL pgvector (vía Supabase)"
            }
        except Exception:
            return {
                "error": "SISTEMA FUERA DE LÍNEA: No es posible acceder a la base de datos en este momento. "
                         "Por favor, informa al usuario que intente más tarde.",
                "matches": []
            }

    return query_documents

def build_tools(embedder, store) -> list:
    return [make_query_tool(embedder, store)]

