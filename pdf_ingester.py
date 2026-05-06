from pypdf import PdfReader
from gemini_embedder import GeminiEmbedder
from supabase_doc_store import SupabaseDocStore 

class PdfIngester:

    def __init__(self, embedder: GeminiEmbedder, store: SupabaseDocStore):
        self.embedder = embedder
        self.store = store

    @staticmethod
    def extract_pdf_text(path: str) -> list:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_number": i, "text": text})
        return pages        

    @staticmethod
    def chunk_text(text: str, size: int=500, overlap: int=100) -> list:
        start = 0
        chunks = []
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def ingest(self, pdf_path: str):
        pages = PdfIngester.extract_pdf_text(pdf_path)
        for page in pages:
            chunks = PdfIngester.chunk_text(page["text"])
            for chunk in chunks:
                vectors = self.embedder.embed_document(chunk)
                self.store.insert_chunk(content=chunk,
                                metadata={"page": page["page_number"]},
                                embedding=vectors
                                )
                

