from google import genai
from google.genai import types
 
class GeminiEmbedder:

    def __init__(self, gemini_api_key: str, model: str = "models/gemini-embedding-001", dims: int=3072):
        self.model = model
        self.dims = dims
        self.client = genai.Client(api_key=gemini_api_key)

    def embed_document(self, text: str) -> list:
        result = self.client.models.embed_content(
            model = self.model,
            contents =  text,
            config = types.EmbedContentConfig(
                task_type = "RETRIEVAL_DOCUMENT",
                output_dimensionality = self.dims
            )
        )
        return list(result.embeddings[0].values)    

    def embed_query(self, text: str) -> list:
        result = self.client.models.embed_content(
            model = self.model,
            contents =  text,
            config = types.EmbedContentConfig(
                task_type = "RETRIEVAL_QUERY",
                output_dimensionality = self.dims
            )
        )
        return list(result.embeddings[0].values)    
