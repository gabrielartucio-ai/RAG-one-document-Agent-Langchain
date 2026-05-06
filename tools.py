from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

def make_query_tool(embedder, store) -> list:
    
    @tool(parse_docstring=True)
    def query_documents(query: str) -> dict:
        """Obtiene información importante y actualizada sobre la política de devoluciones.

        Args:
            query: Una breve consulta de búsqueda semántica en español que captura 
            el tema central de la pregunta del usuario, optimizada para la búsqueda vectorial.
        """

        query_vector = embedder.embed_query(query)
        hits = store.search(query_emb = query_vector)
        result = {
            "query": query,
            "matches": hits,
            "source": "PostgreSQL pgvector"
        }
        return result



#        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
#        def _embed():
#            return embedder.embed_query(query)
        
#        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
#        def _search(vector):
#            return store.search(query_emb=vector)

#        query_vector = _embed()
#        documents = _search(query_vector)
#        return {
#            "context": documents,
#            "source": "PostgreSQL Knowledge Base (pgvector)"
#            }

    return query_documents


def build_tools(embedder, store) -> list:
    return [make_query_tool(embedder, store)]


