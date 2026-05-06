import os
import sys
from dotenv import load_dotenv
from gemini_embedder import GeminiEmbedder
from supabase_doc_store import SupabaseDocStore
from pdf_ingester import PdfIngester

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("The pdf path is required")
        sys.exit(1)

    pdf_path = sys.argv[1]

    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    DATABASE_URL = os.environ.get("DATABASE_URL")

    embedder = GeminiEmbedder(api_key=gemini_api_key)
    store = SupabaseDocStore(database_url=DATABASE_URL)
    pdf_ingester = PdfIngester(embedder=embedder, store=store)

    try:
        pdf_ingester.ingest(pdf_path)
        print(f"Ingesta completada: {pdf_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)