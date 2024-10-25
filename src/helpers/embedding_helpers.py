from openai import AzureOpenAI
from typing import List
import config as cfg

class OpenAIEmbeddingFunction:
    def __init__(self, model: str = "text-embedding-ada-002"):
        client = AzureOpenAI(
            api_key = cfg.API_KEYS['openai_azure'],  
            api_version = "2024-06-01",
            azure_endpoint =cfg.ENDPOINT_AZURE
        )
        self.model = model
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model=self.model)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding