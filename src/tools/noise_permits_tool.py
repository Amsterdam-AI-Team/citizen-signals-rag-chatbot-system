import sys
sys.path.append("..")
import numpy as np
import faiss
import json
from helpers.embedding_helpers import OpenAIEmbeddingFunction
import config as cfg
import mysecrets

from codecarbon import EmissionsTracker


class NoisePermitsTool:
    """
    Fetches and processes noise permits information for the area surrounding a given address.
    """
    def __init__(self, straatnaam: str, huisnummer: str, postcode: str, melding: str):
        self.straatnaam = straatnaam
        self.huisnummer = huisnummer
        self.postcode = postcode
        self.melding = melding

        # Load FAISS index and metadata store for noise permits
        self.index = faiss.read_index(cfg.FAISS_NOISE_PATH)
        with open(cfg.METADATA_STORE_FILE, 'r') as f:
            self.metadata_store = json.load(f)
        
        # Initialize the embedding function
        self.embedder = OpenAIEmbeddingFunction()

    def handle_melding(self, melding: str) -> str:
        """
        Handles a noise melding by searching for similar permits in the FAISS index.
        Based on id of the retrieved index it retreives the actual permit data from the metadata_store json.
        """
        # Embed the melding
        query_embedding = self.embedder.embed_query(melding)
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
    tracker = EmissionsTracker()
    tracker.start()
    # GPU Intensive code goes here
    # Sample values for initializing NoisePermitsTool
    straatnaam = "Zuidplein"
    huisnummer = "136"
    postcode = "1077XV"

    # Sample melding text
    melding_text = "Er is veel bouwlawaai rondom station zuid."

    # Initialize the tool with sample address details
    noise_permit_tool = NoisePermitsTool(straatnaam, huisnummer, postcode, melding_text)

    # Handle the melding and print the result
    result = noise_permit_tool.handle_melding(melding_text)
    print(result)
    tracker.stop()