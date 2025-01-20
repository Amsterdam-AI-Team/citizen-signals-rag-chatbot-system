"""Embedding Helpers"""
from typing import List

from openai import AzureOpenAI

import config as cfg
import my_secrets


class OpenAIEmbeddingFunction:
    """Helper class to embed documents using the AzureOpenAI api"""

    def __init__(self, model: str = "text-embedding-ada-002"):
        client = AzureOpenAI(
            api_key=my_secrets.API_KEYS["openai_azure"],
            api_version="2024-06-01",
            azure_endpoint=cfg.ENDPOINT_AZURE,
        )
        self.model = model
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Given a list of texts, return the corresponding embeddings"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model=self.model)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Given a single query, return the corresponding embedding"""
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
