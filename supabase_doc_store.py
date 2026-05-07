import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

class SupabaseDocStore:

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _get_conn(self):
        conn = psycopg.connect(self.database_url, 
                               row_factory=dict_row,
                               connect_timeout=5)
        register_vector(conn)
        return conn
    
    def insert_chunk(self, content: str, metadata: dict[str, any], embedding: list[float]) -> None:
        sql = """
        insert into public.documents (content, metadata, embedding)
        values (%s, %s, %s)
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (content, Jsonb(metadata), embedding))

    def search(self, query_emb, k: int=5):
        sql = """
        select * from public.match_documents(%s::vector(3072), %s, %s)
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_emb, k, Jsonb({})))
                rows = cur.fetchall()
        return [
            {
                "id": int(r["id"]),
                "content": r["content"],
                "metadata": r["metadata"] or {},
                "similarity": float(r["similarity"])
            }
            for r in rows
        ]
