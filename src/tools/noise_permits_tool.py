import numpy as np
import faiss
import json
from helpers.embedding_helpers import OpenAIEmbeddingFunction
import config as cfg

class NoisePermitsTool:
    """
    Fetches and processes noise permits information for the area surrounding a given address.
    """
    def __init__(self, straatnaam: str, huisnummer: str, postcode: str):
        self.straatnaam = straatnaam
        self.huisnummer = huisnummer
        self.postcode = postcode

        # Load FAISS index and metadata store for noise permits
        self.index = faiss.read_index(cfg.FAISS_NOISE_PATH)
        with open(cfg.METADATA_STORE_PATH, 'r') as f:
            self.metadata_store = json.load(f)
        
        # Initialize the embedding function
        self.embedder = OpenAIEmbeddingFunction()

    def handle_complaint(self, complaint: str) -> str:
        """
        Handles a noise complaint by searching for similar permits in the FAISS index.
        """
        # Embed the complaint
        query_embedding = self.embedder.embed_query(complaint)
        query_embedding_array = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        # Perform similarity search in FAISS
        distances, indices = self.index.search(query_embedding_array, k=1)  # Get the closest match
        if distances[0][0] < 0.8:  # Similarity threshold
            closest_faiss_id = str(indices[0][0])
            permit_data = self.metadata_store.get(closest_faiss_id, "No description available")
            return f"Found matching permit: {permit_data}"
        else:
            return "No matching permit found."

if __name__ == "__main__":
    # Sample values for initializing NoisePermitsTool
    straatnaam = "Zuidplein"
    huisnummer = "136"
    postcode = "1077XV"

    # Sample complaint text
    complaint_text = "Er is veel bouwlawaai rondom station zuid."

    # Initialize the tool with sample address details
    noise_permit_tool = NoisePermitsTool(straatnaam, huisnummer, postcode)

    # Handle the complaint and print the result
    result = noise_permit_tool.handle_complaint(complaint_text)
    print(result)