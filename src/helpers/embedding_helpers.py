from openai import AzureOpenAI
from typing import List
import config as cfg
import my_secrets as sc

class OpenAIEmbeddingFunction:
    def __init__(self, model: str = "text-embedding-ada-002"):
        client = AzureOpenAI(
            api_key = sc.API_KEY,
            api_version = cfg.AZURE_ADA_API_VERSION,
            azure_endpoint =cfg.AZURE_OPENAI_ENDPOINT
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